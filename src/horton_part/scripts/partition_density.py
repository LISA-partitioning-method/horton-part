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


def main():
    """Main program."""
    args = parse_args()
    # Convert the log level string to a logging level
    log_level = getattr(logging, args.log.upper(), None)
    if not isinstance(log_level, int):
        raise ValueError(f"Invalid log level: {args.log}")

    if log_level > logging.DEBUG:
        logging.basicConfig(level=log_level, format="%(levelname)s:    %(message)s")
    else:
        logging.basicConfig(level=log_level, format="%(name)s - %(levelname)s:    %(message)s")

    logger.info("*" * width)
    logger.info(f"Reade grid and density data from {args.filename}")
    logger.info("*" * width)
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

    if args.type in ["gisa", "lisa", "lisa_g"]:
        kwargs["solver"] = args.solver
        kwargs["solver_kwargs"] = {}
        if args.type in ["lisa"]:
            kwargs["basis_func"] = args.func_file or args.basis_func
            if args.solver > 200:
                kwargs["solver_kwargs"]["diis_size"] = args.diis_size

    part = wpart_schemes(args.type)(**kwargs)
    part.do_partitioning()
    # part.do_moments()

    logger.info(" " * width)
    logger.info("*" * width)
    logger.info(" Results ".center(width, " "))
    logger.info("*" * width)
    logger.info("charges:")
    logger.info(part.cache["charges"])
    # logger.info("cartesian multipoles:")
    # logger.info(part.cache["cartesian_multipoles"])
    # logger.info("radial moments:")
    # logger.info(part.cache["radial_moments"])

    if "lisa_g" not in args.type:
        logger.info(" " * width)
        logger.info("*" * width)
        logger.info(" Time usage ".center(width, " "))
        logger.info("*" * width)
        logger.info(
            f"Do Partitioning                              : {part.time_usage['do_partitioning']:>10.2f} s"
        )
        logger.info(
            f"  Update Weights                             : {part._cache['time_update_at_weights']:>10.2f} s"
        )
        logger.info(
            f"    Update Promolecule Density (N_atom**2)   : {part._cache['time_update_promolecule']:>10.2f} s"
        )
        logger.info(
            f"    Update AIM Weights (N_atom)              : {part._cache['time_compute_at_weights']:>10.2f} s"
        )
        logger.info(
            f"  Update Atomic Parameters (iter*N_atom)     : {part._cache['time_update_propars_atoms']:>10.2f} s"
        )
        # logger.info(f"Do Moments                                   : {part.time_usage['do_moments']:>10.2f} s")
        logger.info("*" * width)
        logger.info(" " * width)

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

    if "lisa_g" not in args.type:
        part_data["time"] = part.time_usage["do_partitioning"]
        part_data["time_update_at_weights"] = part._cache["time_update_at_weights"]
        part_data["time_update_promolecule"] = part._cache["time_update_promolecule"]
        part_data["time_compute_at_weights"] = part._cache["time_compute_at_weights"]
        part_data["time_update_propars_atoms"] = part._cache["time_update_propars_atoms"]
        part_data["niter"] = part.cache["niter"]
        part_data["history_charges"] = part.cache["history_charges"]
        part_data["history_propars"] = part.cache["history_propars"]
        part_data["history_entropies"] = part.cache["history_entropies"]

    # part_data["part/cartesian_multipoles"] = part.cache["cartesian_multipoles"]
    # part_data["part/radial_moments"] = part.cache["radial_moments"]
    part_data.update(data)
    np.savez_compressed(args.output, **part_data)


def parse_args():
    """Parse command-line arguments."""
    description = "Molecular density partitioning with HORTON3."
    parser = argparse.ArgumentParser(prog="part-dens", description=description)

    # for part
    parser.add_argument(
        "filename",
        type=str,
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
            "lisa_g",
            "lisa_g_101",
            "lisa_g_104",
            "lisa_g_201",
            "lisa_g_202",
            "lisa_g_206",
            "lisa_g_301",
            "lisa_g_302",
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
        "--solver",
        type=int,
        default=2,
        help="The objective function type for GISA and LISA methods. [default=%(default)s]",
    )
    parser.add_argument(
        "--diis_size",
        type=int,
        default=8,
        help="The number of previous iterations info used in DIIS. [default=%(default)s]",
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

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
