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
import argparse
import os
import sys
import time

import numpy as np
from grid.atomgrid import AtomGrid
from grid.basegrid import OneDGrid
from grid.molgrid import MolGrid

from horton_part import PERIODIC_TABLE, __version__, wpart_schemes

from .program import PartProg

np.set_printoptions(precision=14, suppress=True, linewidth=np.inf)
np.random.seed(44)

__all__ = ["construct_molgrid_from_dict"]


def float_dict(string):
    try:
        # Convert the input string to a dictionary
        # Assuming format "key1_key2:value"
        items = [item.split(":") for item in string.split(",")]
        result = {}
        for item in items:
            key_part = item[0].split("_")
            # Convert key parts to integers
            key = (int(key_part[0]), int(key_part[1]))
            value = float(item[1])
            result[key] = value
        return result
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Each item must be in key1_key2:value format with keys as integers and value as float"
        )


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
        super().__init__("part-dens", width)

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

    def print_propars(self, part, niter=-1, header=None):
        """Print optimized pro-atom parameters.

        The optimized pro-atom parameters are printed out if ISA methods with basis functions are used.
        """
        header = header or "Optimized pro-atom parameters"
        if part.name in ["gisa", "mbis", "lisa", "glisa", "gmbis"]:
            if niter == -np.inf:
                propars = part.initial_propars_modified
            else:
                propars = part.cache["history_propars"][niter, :]
            self.print_header(header)
            for i in range(part.natom):
                begin, end = part._ranges[i], part._ranges[i + 1]
                propars_i = propars[begin:end]
                self.logger.info(f"Propars of No. {i+1} atom {PERIODIC_TABLE[part.numbers[i]]}:")
                if part.name == "mbis":
                    self.logger.info(f"{'Populations':>20}    {'Exponents':>20}")
                    for par, exp in zip(propars_i[::2], propars_i[1::2]):
                        self.logger.info(f"{par:>20.8f}    {exp:>20.8f}")
                elif part.name == "gmbis":
                    self.logger.info(f"{'Populations':>20}    {'Exponents':>20}   {'Orders':>20}")
                    for par, exp, order in zip(propars_i[::3], propars_i[1::3], propars_i[2::3]):
                        self.logger.info(f"{par:>20.8f}    {exp:>20.8f}    {order:>20.8f}")
                else:
                    for par in propars_i:
                        self.logger.info(f"{par:>20.8f}")
                self.logger.info(" ")
            self.print_line()

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

    def single_launch(self, args: argparse.Namespace, fn_in, fn_out, fn_log, **kwargs):
        # Convert the log level string to a logging level
        self.setup_logger(args, fn_log, overwrite=False)
        exclude_keys = []
        if args.type in ["glisa"]:
            exclude_keys = ["inner_threshold", "exp_n"]
        if args.type in ["gisa", "mbis", "isa"]:
            exclude_keys = ["solver", "basis_func", "exp_n"]
        if args.type in ["gmbis"]:
            exclude_keys = ["solver", "basis_func"]
        self.print_settings(args, fn_in, fn_out, fn_log, exclude_keys=exclude_keys)

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
            "lmax": args.lmax,
            "maxiter": args.maxiter,
            "threshold": args.threshold,
            "inner_threshold": args.inner_threshold,
            "radius_cutoff": args.radius_cutoff,
            "logger": self.logger,
        }

        # Arguments for specific methods.
        if args.type in ["gisa", "lisa", "glisa"]:
            if hasattr(args, "solver"):
                part_kwargs["solver"] = args.solver
            if hasattr(args, "solver_options"):
                part_kwargs["solver_options"] = args.solver_options

            if args.type in ["lisa", "glisa"]:
                part_kwargs["basis_func"] = args.func_file or args.basis_func
                if args.type in ["glisa"]:
                    part_kwargs.pop("inner_threshold")
        elif args.type in ["gmbis"]:
            if args.exp_n is None:
                exp_n = {}
            else:
                exp_n = float_dict(args.exp_n)
            part_kwargs["exp_n"] = exp_n

        # Create part object
        part = wpart_schemes(args.type)(**part_kwargs)
        try:
            getattr(part, args.part_job_type)()
            # part.do_partitioning()
        except RuntimeError as e:
            self.logger.info(e)
            return 1

        # part.do_moments()

        # Print results
        self.print_basis(part)
        if args.type in ["gisa", "lisa", "glisa"]:
            self.print_propars(part, -np.inf, "The modified initial propars")
        self.print_header("Results")
        self.print_charges(data["atnums"], part.cache["charges"])
        # logger.info("cartesian multipoles:")
        # logger.info(part.cache["cartesian_multipoles"])
        # logger.info("radial moments:")
        # logger.info(part.cache["radial_moments"])
        self.print_propars(part)
        self.print_part_time(part)

        # Collect partition data
        part_data = {
            # inputs
            "natom": len(data["atnums"]),
            "atnums": data["atnums"],
            "atcorenums": data["atcorenums"],
            "type": args.type,
            "lmax": args.lmax,
            "maxiter": args.maxiter,
            "threshold": args.threshold,
            "inner_threshold": args.inner_threshold if args.type not in ["glisa"] else np.nan,
            "solver": args.solver,
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

        # part_data["part/cartesian_multipoles"] = part.cache["cartesian_multipoles"]
        # part_data["part/radial_moments"] = part.cache["radial_moments"]

        if args.part_job_type == "do_density_decomposition":
            bs_info = self.load_basis_info(part)
            propars = part.cache["history_propars"][-1, :]
            for iatom in range(part.natom):
                for k in [
                    f"radial_points_{iatom}",
                    f"spherical_average_{iatom}",
                    f"radial_weights_{iatom}",
                ]:
                    part_data[k] = part.cache[k]
                # TODO: use optimized results
                propars_a = propars[part._ranges[iatom] : part._ranges[iatom + 1]]
                if args.type in ["gisa", "lisa", "glisa"]:
                    info = np.asarray(bs_info[part.numbers[iatom]]).T
                    info[:, -1] = propars_a
                elif args.type in ["mbis"]:
                    propars_a = propars_a.reshape((-1, 2))
                    info = np.ones((propars_a.shape[0], 3))
                    info[:, 1] = propars_a[:, 1]
                    info[:, 2] = propars_a[:, 0]
                elif args.type in ["gmbis"]:
                    propars_a = propars_a.reshape((-1, 3))
                    info = np.zeros((propars_a.shape[0], 3))
                    info[:, 0] = propars_a[:, 2]
                    info[:, 1] = propars_a[:, 1]
                    info[:, 2] = propars_a[:, 0]
                else:
                    raise NotImplementedError
                part_data[f"bs_info_{iatom}"] = info

        # NOTE: do not restore molecular density and grids
        # part_data.update(data)
        path = os.path.dirname(os.path.abspath(fn_out))
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez_compressed(fn_out, **part_data)
        return 0

    def build_parser(self):
        """Parse command-line arguments."""
        description = f"Molecular density partitioning with HORTON3 {__version__}."
        parser = argparse.ArgumentParser(prog="part-dens", description=description)

        # for part
        parser.add_argument(
            "--inputs",
            type=str,
            nargs="+",
            default=None,
            help="The output file of part-gen command.",
        )
        parser.add_argument(
            "-t",
            "--type",
            type=str,
            default="lisa",
            choices=["gisa", "lisa", "mbis", "is", "glisa", "gmbis"],
            help="Number of angular grid points. [default=%(default)s]",
        )
        parser.add_argument(
            "--basis_func",
            type=str,
            default="gauss",
            choices=["gauss", "slater"],
            help="The type of basis functions. [default=%(default)s]",
        )
        parser.add_argument(
            "--func_file",
            type=str,
            default=None,
            help="The json inputs of basis functions.",
        )
        parser.add_argument(
            "--maxiter",
            type=int,
            default=1000,
            help="The maximum outer iterations. [default=%(default)s]",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=1e-6,
            help="The threshold of convergence for outer iterations. [default=%(default)s]",
        )
        parser.add_argument(
            "--inner_threshold",
            type=float,
            default=1e-8,
            help="The inner threshold of convergence for local version methods. [default=%(default)s]",
        )
        parser.add_argument(
            "--lmax",
            type=int,
            default=3,
            help="The maximum angular momentum in multipole expansions. [default=%(default)s]",
        )
        parser.add_argument(
            "--radius_cutoff",
            type=float,
            default=np.inf,
            help="The radius cutoff of local atomic grid [default=%(default)s]",
        )
        parser.add_argument(
            "--outputs",
            help="The NPZ file in which the partitioning results will be stored.",
            type=str,
            nargs="+",
            default="partitioning.npz",
        )
        parser.add_argument(
            "--log_level",
            default="INFO",
            choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Set the logging level (default: %(default)s)",
        )
        parser.add_argument(
            "--log_files",
            type=str,
            nargs="+",
            default=None,
            help="The log file.",
        )
        parser.add_argument(
            "--solver",
            type=str,
            default=None,
            help="The objective function type for GISA and LISA methods. [default=%(default)s]",
        )
        parser.add_argument(
            "--exp_n",
            type=float_dict,
            default=None,
            help="The exponent of radial distance used in the Generalize MBIS method. [default=%(default)s]",
        )
        parser.add_argument(
            "--part_job_type",
            type=str,
            default="do_partitioning",
            help="The type of partitioning job. [default=%(default)s]",
        )
        parser.add_argument(
            "--config_file",
            type=str,
            default=None,
            help="Use configure file.",
        )
        return parser


def main(args=None) -> int:
    """Main entry."""
    return PartDensProg().run(args)


if __name__ == "__main__":
    sys.exit(main())
