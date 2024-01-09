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
import logging
import sys

import numpy as np
from grid.atomgrid import AtomGrid
from grid.basegrid import OneDGrid
from grid.molgrid import MolGrid

from horton_part import wpart_schemes

from .generate_density import load_settings_from_yaml_file

width = 100
np.set_printoptions(precision=14, suppress=True, linewidth=np.inf)
np.random.seed(44)

__all__ = ["construct_molgrid_from_dict"]


logger = logging.getLogger(__name__)


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


def main(args=None) -> int:
    """Main program."""
    parser = build_parser()
    args = parser.parse_args(args)
    args = load_settings_from_yaml_file(args, "part-dens")

    if args.filename is None:
        parser.print_help()
        return 0

    # Convert the log level string to a logging level
    log_level = getattr(logging, args.log.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log}")

    if log_level > logging.DEBUG:
        logging.basicConfig(level=log_level, format="%(message)s")
    else:
        logging.basicConfig(level=log_level, format="%(name)s - %(levelname)s:    %(message)s")

    logger.info("*" * width)
    logger.info("Settings for part-dens program".center(width, " "))
    logger.info("*" * width)
    for k, v in vars(args).items():
        logger.info(f"{k:>10} : {str(v):<10}".center(width, " "))
    logger.info("-" * width)
    logger.info(" ")

    # logger.info("*" * width)
    logger.info(f"Reade grid and density data from {args.filename} ...")
    # logger.info("*" * width)
    data = np.load(args.filename)
    grid = construct_molgrid_from_dict(data)

    logger.info(" " * width)
    logger.info("*" * width)
    logger.info(" Partitioning ".center(width, " "))
    logger.info("*" * width)
    kwargs = {
        "coordinates": data["atcoords"],
        "numbers": data["atnums"],
        "pseudo_numbers": data["atcorenums"],
        "grid": grid,
        "moldens": data["density"],
        "lmax": args.lmax,
        "maxiter": args.maxiter,
        "threshold": args.threshold,
        "radius_cutoff": args.radius_cutoff,
    }

    if args.type in ["gisa", "lisa"]:
        kwargs["solver"] = args.solver
        if hasattr(args, "solver_options"):
            kwargs["solver_options"] = args.solver_options

    if args.type in ["lisa"] or "glisa" in args.type:
        kwargs["basis_func"] = args.func_file or args.basis_func

    part = wpart_schemes(args.type)(**kwargs)
    part.do_partitioning()
    # part.do_moments()

    logger.info(" " * width)
    logger.info("*" * width)
    logger.info(" Results ".center(width, " "))
    logger.info("*" * width)
    logger.info("  Atomic charges [a.u.]:")

    for atnum, chg in zip(data["atnums"], part.cache["charges"]):
        logger.info(f"{atnum:>4} : {chg:>15.8f}".center(width, " "))
    logger.info("-" * width)
    logger.info(" " * width)
    # logger.info("cartesian multipoles:")
    # logger.info(part.cache["cartesian_multipoles"])
    # logger.info("radial moments:")
    # logger.info(part.cache["radial_moments"])

    if "lisa_g" not in args.type:
        logger.info("*" * width)
        logger.info(" Time usage ".center(width, " "))
        logger.info("*" * width)
        logger.info(f"{'Do Partitioning':<45} : {part.time_usage['do_partitioning']:>10.2f} s")
        logger.info(f"{'  Update Weights':<45} : {part.cache['time_update_at_weights']:>10.2f} s")
        logger.info(
            f"{'    Update Promolecule Density (N_atom**2)':<45} : {part.cache['time_update_promolecule']:>10.2f} s"
        )
        logger.info(
            f"{'    Update AIM Weights (N_atom)':<45} : {part.cache['time_compute_at_weights']:>10.2f} s"
        )
        logger.info(
            f"{'  Update Atomic Parameters (iter*N_atom)':<45} : {part.cache['time_update_propars_atoms']:>10.2f} s"
        )
        # logger.info(f"Do Moments                                   : {part.time_usage['do_moments']:>10.2f} s")
        logger.info("-" * width)

    part_data = {}
    part_data["natom"] = len(data["atnums"])
    part_data["atnums"] = data["atnums"]
    part_data["atcorenums"] = data["atcorenums"]
    part_data["type"] = args.type
    part_data["lmax"] = args.lmax
    part_data["maxiter"] = args.maxiter
    part_data["threshold"] = args.threshold
    part_data["solver"] = args.solver
    part_data["charges"] = part.cache["charges"]

    if "glisa" not in args.type:
        part_data["time"] = part.time_usage["do_partitioning"]
        part_data["time_update_at_weights"] = part.cache["time_update_at_weights"]
        part_data["time_update_promolecule"] = part.cache["time_update_promolecule"]
        part_data["time_compute_at_weights"] = part.cache["time_compute_at_weights"]
        part_data["time_update_propars_atoms"] = part.cache["time_update_propars_atoms"]
        part_data["niter"] = part.cache["niter"]
        part_data["history_charges"] = part.cache["history_charges"]
        part_data["history_propars"] = part.cache["history_propars"]
        part_data["history_entropies"] = part.cache["history_entropies"]

    # part_data["part/cartesian_multipoles"] = part.cache["cartesian_multipoles"]
    # part_data["part/radial_moments"] = part.cache["radial_moments"]
    part_data.update(data)
    np.savez_compressed(args.output, **part_data)
    return 0


def build_parser():
    """Parse command-line arguments."""
    description = "Molecular density partitioning with HORTON3."
    parser = argparse.ArgumentParser(prog="part-dens", description=description)

    # for part
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        default=None,
        help="The output file of part-gen command.",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="lisa",
        choices=[
            "gisa",
            "lisa",
            "mbis",
            "is",
            "glisa-cvxopt",
            "glisa-trust-constr",
            "glisa-sc",
            "glisa-diis",
            "glisa-cdiis",
        ],
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
        help="The json filename of basis functions.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=1000,
        help="The maximum iterations. [default=%(default)s]",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-6,
        help="The threshold of convergence. [default=%(default)s]",
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
        "--output",
        help="The NPZ file in which the partitioning results will be stored.",
        type=str,
        default="partitioning.npz",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["NOTSET", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: %(default)s)",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="cvxopt",
        help="The objective function type for GISA and LISA methods. [default=%(default)s]",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Use configure file.",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main())
