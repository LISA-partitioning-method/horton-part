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

from horton_part import wpart_schemes

from .program import PartProg

np.set_printoptions(precision=14, suppress=True, linewidth=np.inf)
np.random.seed(44)

__all__ = ["construct_molgrid_from_dict"]


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
            f"{'    Update Promolecule Density (N_atom**2)':<45} : {part.cache['time_update_promolecule']:>10.2f} s"
        )
        self.logger.info(
            f"{'    Update AIM Weights (N_atom)':<45} : {part.cache['time_compute_at_weights']:>10.2f} s"
        )
        self.logger.info(
            f"{'  Update Atomic Parameters (iter*N_atom)':<45} : {part.cache['time_update_propars_atoms']:>10.2f} s"
        )
        # logger.info(f"Do Moments                                   : {part.time_usage['do_moments']:>10.2f} s")
        self.print_line()

    def single_launch(self, args: argparse.Namespace, fn_in, fn_out, fn_log):
        # Convert the log level string to a logging level
        self.setup_logger(args, fn_log, overwrite=False)
        exclude_keys = []
        if args.type in ["glisa"]:
            exclude_keys = ["inner_threshold"]
        if args.type not in ["gisa", "lisa", "glisa"]:
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

        # Create part object
        part = wpart_schemes(args.type)(**part_kwargs)
        try:
            part.do_partitioning()
        except RuntimeError as e:
            self.logger.info(e)
            return 1

        # part.do_moments()

        # Print results
        self.print_header("Results")
        self.print_charges(data["atnums"], part.cache["charges"])
        # logger.info("cartesian multipoles:")
        # logger.info(part.cache["cartesian_multipoles"])
        # logger.info("radial moments:")
        # logger.info(part.cache["radial_moments"])
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
            "inner_threshold": args.threshold,
            "solver": args.solver,
            # results
            "charges": part.cache["charges"],
            "time": part.time_usage["do_partitioning"],
            "time_update_at_weights": part.cache["time_update_at_weights"],
            "time_update_promolecule": part.cache["time_update_promolecule"],
            "time_compute_at_weights": part.cache["time_compute_at_weights"],
            "time_update_propars_atoms": part.cache["time_update_propars_atoms"],
            "niter": part.cache["niter"],
            "history_charges": part.cache["history_charges"],
            "history_propars": part.cache["history_propars"],
            "history_entropies": part.cache["history_entropies"],
        }

        # part_data["part/cartesian_multipoles"] = part.cache["cartesian_multipoles"]
        # part_data["part/radial_moments"] = part.cache["radial_moments"]
        part_data.update(data)
        path = os.path.dirname(os.path.abspath(fn_out))
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez_compressed(fn_out, **part_data)
        return 0

    def build_parser(self):
        """Parse command-line arguments."""
        description = "Molecular density partitioning with HORTON3."
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
            choices=["gisa", "lisa", "mbis", "is", "glisa"],
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
