# HORTON-PART: molecular density partition schemes based on HORTON package.
# Copyright (C) 2023-2024 The HORTON-PART Development Team
#
# This file is part of HORTON-PART
#
# HORTON-PART is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON-PART is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
import os
import sys
import time

import numpy as np
import yaml
from grid.atomgrid import AtomGrid
from grid.basegrid import OneDGrid
from grid.molgrid import MolGrid

from horton_part import PERIODIC_TABLE, __version__, wpart_schemes
from horton_part.utils import DATA_PATH

from .program import PartProg

np.set_printoptions(precision=14, suppress=True, linewidth=np.inf)
np.random.seed(44)

__all__ = ["construct_molgrid_from_dict"]


def get_nested_attr(obj, attr_path):
    """
    Recursively get an attribute from an object using dot notation.
    """
    attrs = attr_path.split(".")
    for attr in attrs:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def prepare_exp_n_dict(dict):
    """Prepare a dict including exponent and order."""
    res = {}
    if dict is None:
        return res

    for Z, exp_list in dict.items():
        for ishell, value in enumerate(exp_list):
            res[(int(Z), ishell)] = float(value)
    return res


def construct_molgrid_from_dict(data):
    atcoords = data["atcoords"]
    atnums = data["atnums"]
    # atcorenums = data["atcorenums"]
    aim_weights = data["aim_weights"]
    natom = len(atnums)

    # build atomic grids
    atgrids = []
    for iatom in range(natom):
        rgrid = OneDGrid(
            points=data[f"atom{iatom}/rgrid/points"],
            weights=data[f"atom{iatom}/rgrid/weights"],
        )
        shell_idxs = data[f"atom{iatom}/shell_idxs"]
        sizes = shell_idxs[1:] - shell_idxs[:-1]
        # center = atcoords[iatom]
        atgrid = AtomGrid(rgrid, sizes=sizes, center=atcoords[iatom], rotate=0)
        atgrids.append(atgrid)

    return MolGrid(atnums, atgrids, aim_weights=aim_weights, store=True)


class PartDensProg(PartProg):
    """Part-Dens Program"""

    def __init__(self, width=100):
        description = f"Molecular density partitioning with HORTON3 {__version__}."
        super().__init__("part-dens", width, description=description)

    def print_part_time(self, part):
        """Print partitioning time usage info."""
        self.print_header("Time usage")
        self.logger.info(f"{'Do Partitioning':<45} : {part.time_usage['do_partitioning']:>10.2f} s")
        self.logger.info(
            f"{'  Update Weights':<45} : {part.cache['time_update_at_weights']:>10.2f} s"
        )
        self.logger.info(
            f"{'  Update Atomic Parameters (iter*N_atom)':<45} : {part.cache['time_update_propars']:>10.2f} s"
        )
        # logger.info(f"Do Moments                                   : {part.time_usage['do_moments']:>10.2f} s")
        self.print_line()

    # TODO: put the print_propars to each method.
    def print_propars(self, part, niter=-1, header=None):
        """Print optimized pro-atom parameters.

        The optimized pro-atom parameters are printed out if ISA methods with basis functions are used.
        """
        header = header or "Optimized pro-atom parameters"
        if part.name in ["gisa", "mbis", "lisa", "glisa", "gmbis", "nlis"]:
            if niter == -np.inf:
                propars = part.initial_propars_modified
            else:
                propars = part.cache["history_propars"][niter, :]
            self.print_header(header)
            for i in range(part.natom):
                begin, end = part._ranges[i], part._ranges[i + 1]
                propars_i = propars[begin:end]
                self.logger.info(f"Propars of No. {i+1} atom {PERIODIC_TABLE[part.numbers[i]]}:")
                if part.name in ["mbis"]:
                    self.logger.info(f"{'Populations':>20}    {'Exponents':>20}")
                    for par, exp in zip(propars_i[::2], propars_i[1::2]):
                        self.logger.info(f"{par:>20.8f}    {exp:>20.8f}")
                elif part.name in ["gmbis", "nlis"]:
                    self.logger.info(f"{'Populations':>20}    {'Exponents':>20}   {'Orders':>20}")
                    for par, exp, order in zip(propars_i[::3], propars_i[1::3], propars_i[2::3]):
                        self.logger.info(f"{par:>20.8f}    {exp:>20.8f}    {order:>20.8f}")
                else:
                    for par in propars_i:
                        self.logger.info(f"{par:>20.8f}")
                self.logger.info(" ")
            self.print_line()

    # TODO: also put this function to each part method
    def load_basis_info(self, part):
        if part.name not in ["gisa", "lisa", "glisa"]:
            return {}

        res = {}
        numbers = sorted(set(part.numbers))
        bs_helper = part.bs_helper
        for number in numbers:
            orders = bs_helper.orders[number]
            exponents = bs_helper.exponents[number]
            initials = bs_helper.initials[number]
            res[number] = [orders, exponents, initials]
        return res

    def print_basis(self, part):
        """Print basis functions used in Partitioning methods."""

        if part.name in ["gisa", "lisa", "glisa"]:
            bs_info = self.load_basis_info(part)

            self.print_header("Basis functions")
            self.logger.info("    Exponential order  Exponents coefficients  Initials values")
            self.logger.info("    -----------------  ----------------------  ---------------")
            numbers = sorted(set(part.numbers))
            for number in numbers:
                self.logger.info(f"Atom {PERIODIC_TABLE[number]}")
                for n, cak, pop in zip(*bs_info[number]):
                    self.logger.info(f"{str(n):>8}          {cak:>15.6f}         {pop:>15.6f}")
            self.print_line()

    def single_launch(self, settings, fn_in, fn_out, fn_log, **kwargs):
        self.setup_logger(settings, fn_log, overwrite=False)
        type = settings.get("type")
        with open(DATA_PATH / "keywords.yaml") as f:
            keywords = yaml.safe_load(f)
        exclude_keys = [
            k
            for k in settings.keys()
            if k not in keywords[type]["io_infos"] + keywords[type]["class_args"]
        ]
        self.print_settings(settings, fn_in, fn_out, fn_log, exclude_keys=exclude_keys)

        self.logger.info(f"Load grid and density data from {fn_in} ...")
        t0 = time.time()
        data = np.load(fn_in)
        t1 = time.time()
        self.logger.info("Load data done!")
        self.logger.info(f"Time usage {t1 - t0:.3f} s")

        self.logger.info("Create molecular grid ...")
        t0 = time.time()
        grid = construct_molgrid_from_dict(data)
        t1 = time.time()
        self.logger.info("Create molecular grid done!")
        self.logger.info(f"Time usage {t1 - t0:.3f} s")

        self.print_header("Partitioning")

        # Basic arguments required to create a `Part` object.
        part_kwargs = {
            "coordinates": data["atcoords"],
            "numbers": data["atnums"],
            "pseudo_numbers": data["atcorenums"],
            "grid": grid,
            "moldens": data["density"],
            "logger": self.logger,
        }

        # Arguments for specific methods.
        for k in keywords[type]["class_args"]:
            if k in settings:
                part_kwargs[k] = settings[k]

        if "basis_func" in keywords[type]["class_args"]:
            part_kwargs["basis_func"] = settings.get("func_file") or settings.get("basis_func")

        if "exp_n_dict" in keywords[type]["class_args"]:
            part_kwargs["exp_n_dict"] = prepare_exp_n_dict(settings.get("exp_n_dict", None))

        if "nshell_dict" in keywords[type]["class_args"]:
            part_kwargs["nshell_dict"] = settings.get("nshell_dict", None)

        # Create part object
        part = wpart_schemes(type)(**part_kwargs)
        try:
            getattr(part, settings["part_job_type"])()
        except RuntimeError as e:
            self.logger.info(e)
            return 1

        # Print results
        self.print_basis(part)
        if type in ["gisa", "lisa", "glisa"]:
            self.print_propars(part, -np.inf, "The modified initial propars")
        self.print_header("Results")
        self.print_charges(data["atnums"], part.cache["charges"])
        self.print_propars(part)
        self.print_part_time(part)

        # Collect partition data
        part_data = {
            # inputs
            "natom": len(data["atnums"]),
            "atnums": data["atnums"],
            "atcorenums": data["atcorenums"],
            "type": type,
            "lmax": settings["lmax"],
            "maxiter": settings["maxiter"],
            "threshold": settings["threshold"],
            "inner_threshold": settings["inner_threshold"] if type not in ["glisa"] else np.nan,
            "solver": settings["solver"],
            # results
            "charges": part.cache["charges"],
            "time": part.time_usage["do_partitioning"],
            "time_update_at_weights": part.cache["time_update_at_weights"],
            "time_update_propars": part.cache["time_update_propars"],
            "niter": part.cache["niter"],
            "history_charges": part.cache["history_charges"],
            "history_propars": part.cache["history_propars"],
            "history_entropies": part.cache["history_entropies"],
            "history_changes": part.cache["history_changes"],
        }

        # TODO: compute molecular quadrupole moment from molecular density.
        # part_data["atcoords"] = part.coordinates
        # part_data["part/cartesian_multipoles"] = part.cache["cartesian_multipoles"]
        # part_data["part/radial_moments"] = part.cache["radial_moments"]

        # coords = part.grid.points
        # x = coords[:, 0]
        # y = coords[:, 1]
        # z = coords[:, 2]
        # r2 = x**2 + y**2 + z**2
        # # Note: numerical issue is problematic for quadrupole moment.
        # Qxx = part.grid.integrate(-1 / 2 * part._moldens, 3 * x**2 - r2)
        # Qyy = part.grid.integrate(-1 / 2 * part._moldens, 3 * y**2 - r2)
        # Qzz = part.grid.integrate(-1 / 2 * part._moldens, 3 * z**2 - r2)

        # Qxy = part.grid.integrate(-1 / 2 * part._moldens, 3 * x * y)
        # Qxz = part.grid.integrate(-1 / 2 * part._moldens, 3 * x * z)
        # Qyz = part.grid.integrate(-1 / 2 * part._moldens, 3 * y * z)
        # Q = np.array([[Qxx, Qxy, Qxz], [Qxy, Qyy, Qyz], [Qxz, Qyz, Qzz]])
        # part_data["quadrupole_moment"] = Q

        # Dx = part.grid.integrate(-1 * part._moldens, x)
        # Dy = part.grid.integrate(-1 * part._moldens, y)
        # Dz = part.grid.integrate(-1 * part._moldens, z)
        # D = np.array([Dx, Dy, Dz])
        # part_data["dipole_moment"] = D

        if settings["part_job_type"] == "do_density_decomposition":
            # For each part job, we can store different properties.
            bs_info = self.load_basis_info(part)
            propars = part.cache["history_propars"][-1, :]
            for iatom in range(part.natom):
                for k in [
                    f"radial_points_{iatom}",
                    f"spherical_average_{iatom}",
                    f"radial_weights_{iatom}",
                ]:
                    part_data[k] = part.cache[k]
                propars_a = propars[part._ranges[iatom] : part._ranges[iatom + 1]]
                if type in ["gisa", "lisa", "glisa"]:
                    info = np.asarray(bs_info[part.numbers[iatom]]).T
                    info[:, -1] = propars_a
                elif type in ["mbis"]:
                    propars_a = propars_a.reshape((-1, 2))
                    info = np.ones((propars_a.shape[0], 3))
                    info[:, 1] = propars_a[:, 1]
                    info[:, 2] = propars_a[:, 0]
                elif type in ["gmbis", "nlis"]:
                    propars_a = propars_a.reshape((-1, 3))
                    info = np.zeros((propars_a.shape[0], 3))
                    info[:, 0] = propars_a[:, 2]
                    info[:, 1] = propars_a[:, 1]
                    info[:, 2] = propars_a[:, 0]
                elif type in ["is"]:
                    info = propars_a
                else:
                    raise NotImplementedError
                part_data[f"bs_info_{iatom}"] = info

        if settings.get("save"):
            self.print_header("Cache")
            self.logger.info(f"The following extra infos are stored in {fn_out}:")
            for _k in settings["save"]:
                # Check for dot notation (nested attributes)
                if isinstance(_k, list):
                    _k = tuple(_k)
                    value = None
                else:
                    value = get_nested_attr(part, _k)
                if isinstance(value, np.ndarray):
                    self.logger.info(f"{str(_k):>40} => save/part.{_k}")
                    part_data[f"save/part.{_k}"] = value
                if value is None and _k in part.cache:
                    self.logger.info(f"{str(_k):>40} => save/part.cache/{_k}")
                    part_data[f"save/part.cache/{_k}"] = part.cache[_k]
            self.print_line()

        # NOTE: do not restore molecular density and grids
        # part_data.update(data)
        path = os.path.dirname(os.path.abspath(fn_out))
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez_compressed(fn_out, **part_data)
        return 0


def main(args=None) -> int:
    """Main entry."""
    return PartDensProg().run(args)


if __name__ == "__main__":
    sys.exit(main())
