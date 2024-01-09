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
import os
import sys

import numpy as np
import yaml
from gbasis.evals.eval import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis
from gbasis.wrappers import from_iodata
from grid.becke import BeckeWeights
from grid.molgrid import MolGrid
from grid.onedgrid import GaussChebyshev
from grid.rtransform import BeckeRTransform
from iodata import load_one

from horton_part.core.logging import setup_logger

width = 100
np.set_printoptions(precision=14, suppress=True, linewidth=np.inf)
np.random.seed(44)

__all__ = [
    "prepare_input",
    "load_settings_from_yaml_file",
    "print_settings",
]

logger = logging.getLogger(__name__)


def prepare_input(iodata, nrad, nang, chunk_size, gradient, orbitals, store_atgrids):
    """Prepare input for denspart with HORTON3 modules.

    Parameters
    ----------
    iodata
        An instance with IOData containing the necessary data to compute the
        electron density on the grid.
    nrad
        Number of radial grid points.
    nang
        Number of angular grid points.
    chunk_size
        Number of points on which the density is evaluated in one pass.
    gradient
        When True, also the gradient of the density is computed.
    orbitals
        When True, also the occupied and virtual orbitals are computed.
    store_atgrids
        When True, the atomic grids are also stored.

    Returns
    -------
    grid
        A molecular integration grid.
    data
        Qauntities evaluated on the grid, includeing the density.

    """
    grid = _setup_grid(iodata.atnums, iodata.atcoords, nrad, nang, store_atgrids)
    data = _compute_stuff(iodata, grid.points, gradient, orbitals, chunk_size)
    return grid, data


# pylint: disable=protected-access
def _setup_grid(atnums, atcoords, nrad, nang, store_atgrids):
    """Set up a simple molecular integration grid for a given molecular geometry.

    Parameters
    ----------
    atnums: np.ndarray(N,)
        Atomic numbers
    atcoords: np.ndarray(N, 3)
        Atomic coordinates.
    store_atgrids
        When True, the atomic grids are also stored.

    Returns
    -------
    grid
        A molecular integration grid, instance (of a subclass of)
        grid.basegrid.Grid.

    """
    logger.info("Setting up grid")
    becke = BeckeWeights(order=3)
    # Fix for missing radii.
    becke._radii[2] = 0.5
    becke._radii[10] = 1.0
    becke._radii[18] = 2.0
    becke._radii[36] = 2.5
    becke._radii[54] = 3.5
    oned = GaussChebyshev(nrad)
    rgrid = BeckeRTransform(1e-4, 1.5).transform_1d_grid(oned)
    grid = MolGrid.from_size(atnums, atcoords, rgrid, nang, becke, store=store_atgrids, rotate=0)
    assert np.isfinite(grid.points).all()
    assert np.isfinite(grid.weights).all()
    assert (grid.weights >= 0).all()
    # TODO: remove grid points with zero weight
    return grid


def _compute_stuff(iodata, points, gradient, orbitals, chunk_size):
    """Evaluate the density and other things on a give set of grid points.

    Parameters
    ----------
    iodata: IOData
        An instance of IOData, containing an atomic orbital basis set.
    points: np.ndarray(N, 3)
        A set of grid points.
    chunk_size
        Number of points on which the density is evaluated in one pass.
    gradient
        When True, also the gradient of the density is computed.
    orbitals
        When True, also the occupied and virtual orbitals are computed.

    Returns
    -------
    results
        Dictionary with density and optionally other quantities.

    """
    one_rdm = iodata.one_rdms.get("post_scf", iodata.one_rdms.get("scf"))
    if one_rdm is None:
        if iodata.mo is None:
            raise ValueError(
                "The input file lacks wavefunction data with which " "the density can be computed."
            )
        coeffs, occs = iodata.mo.coeffs, iodata.mo.occs
        one_rdm = np.dot(coeffs * occs, coeffs.T)
    basis, coord_types = from_iodata(iodata)

    # Prepare result dictionary.
    result = {"density": np.zeros(len(points))}
    if gradient:
        result["density_gradient"] = np.zeros((len(points), 3))
    if orbitals:
        if iodata.mo is None:
            raise ValueError("No orbitals found in file.")
        # TODO: generalize code towards other kinds of orbitals.
        if iodata.mo.kind != "restricted":
            raise NotImplementedError("Only restricted orbitals are supported.")
        result["mo_occs"] = iodata.mo.occs
        result["mo_energies"] = iodata.mo.energies
        result["orbitals"] = np.zeros((len(points), iodata.mo.norb))
        if gradient:
            result["orbitals_gradient"] = np.zeros((len(points), iodata.mo.norb, 3))

    # Actual computation in chunks.
    istart = 0
    while istart < len(points):
        iend = min(istart + chunk_size, len(points))
        logger.info(f"Computing stuff: {istart} ... {iend} / {len(points)}")
        # Basis functions are computed upfront for efficiency.
        logger.info("  basis")
        basis_grid = evaluate_basis(basis, points[istart:iend], coord_type=coord_types)
        if gradient:
            logger.info("  basis_gradient")
            basis_gradient_grid = np.array(
                [
                    evaluate_deriv_basis(basis, points[istart:iend], orders, coord_type=coord_types)
                    for orders in np.identity(3, dtype=int)
                ]
            )
        # Use basis functions on grid for various quantities.
        logger.info("  density")
        result["density"][istart:iend] = np.einsum(
            "ab,bp,ap->p", one_rdm, basis_grid, basis_grid, optimize=True
        )
        if gradient:
            logger.info("  density gradient")
            result["density_gradient"][istart:iend] = 2 * np.einsum(
                "ab,bp,cap->pc", one_rdm, basis_grid, basis_gradient_grid, optimize=True
            )
        if orbitals:
            logger.info("  orbitals")
            result["orbitals"][istart:iend] = np.einsum("bo,bp->po", iodata.mo.coeffs, basis_grid)
            if gradient:
                logger.info("  orbitals gradient")
                result["orbitals_gradient"][istart:iend] = np.einsum(
                    "bo,cbp->poc", iodata.mo.coeffs, basis_gradient_grid
                )
        istart = iend
    assert (result["density"] >= 0).all()
    return result


def load_settings_from_yaml_file(args, sub_cmd="part-gen", fn_key="config_file"):
    """
    Load settings from a YAML configuration file and update the 'args' object.

    This function reads a YAML file specified by the 'fn_key' attribute of the 'args' object.
    It then updates 'args' with the settings found under the specified 'cmd' section of the YAML file.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments object, typically obtained from argparse. This object is updated
        with the settings from the YAML file.
    sub_cmd : str, optional
        The key within the YAML file that contains the settings to be loaded. Only settings
        under this key are used to update 'args'. Default is 'part-gen'.
    fn_key : str, optional
        The attribute name in 'args' that holds the path to the YAML configuration file.
        Default is 'config_file'.

    Returns
    -------
    argparse.Namespace
        The updated arguments object with settings loaded from the YAML file.

    Raises
    ------
    AssertionError
        If 'args' does not have an attribute named 'fn_key', or if the 'cmd' key is not
        found in the loaded YAML settings.

    Notes
    -----
    - The function asserts that the 'args' object has an attribute named as per 'fn_key'.
    - It also checks if the specified YAML file exists before attempting to open it.
    - Settings are loaded only if the 'cmd' key is present in the YAML file.
    - Each setting under the 'cmd' section in the YAML file updates the corresponding attribute
      in the 'args' object.
    """
    assert hasattr(args, fn_key)
    yaml_fn = getattr(args, fn_key)
    if yaml_fn and os.path.exists(yaml_fn):
        with open(yaml_fn) as f:
            settings = yaml.safe_load(f)
        if sub_cmd:
            assert sub_cmd in settings
            for k, v in settings[sub_cmd].items():
                setattr(args, k, v)
        else:
            for k, v in settings.items():
                setattr(args, k, v)
    return args


def main(args=None) -> int:
    """Main program."""
    parser = build_parser()
    args = parser.parse_args(args)
    args = load_settings_from_yaml_file(args)
    if args.inputs is None:
        parser.print_help()
        return 0

    log_files = getattr(args, "log_files", None)
    inputs = args.inputs
    outputs = args.outputs
    if not isinstance(inputs, list):
        inputs = [inputs]
        outputs = [outputs]
    if log_files is None:
        log_files = [None] * len(inputs)
    assert len(inputs) == len(outputs) == len(log_files)
    for fn_in, fn_out, fn_log in zip(inputs, outputs, log_files):
        failed = sub_main(args, fn_in, fn_out, fn_log)
        if failed:
            return 1
    else:
        return 0


def print_settings(logger, args, cmd_name, fn_in, fn_out, fn_log):
    logger.info("*" * width)
    logger.info(f"Settings for {cmd_name} program".center(width, " "))
    logger.info("*" * width)
    for k, v in vars(args).items():
        if k in ["inputs", "outputs", "log_files"]:
            if k == "inputs":
                logger.info(f"{'input':>40} : {str(fn_in):<40}".center(width, " "))
            elif k == "outputs":
                logger.info(f"{'output':>40} : {str(fn_out):<40}".center(width, " "))
            else:
                logger.info(f"{'log_file':>40} : {str(fn_log):<40}".center(width, " "))

        else:
            logger.info(f"{k:>40} : {str(v):<40}".center(width, " "))
    logger.info("-" * width)
    logger.info(" ")


def sub_main(args, fn_in, fn_out, fn_log):
    # Convert the log level string to a logging level
    log_level = getattr(logging, args.log_level.upper(), None)
    setup_logger(logger, log_level, fn_log)
    print_settings(logger, args, "part-gen", fn_in, fn_out, fn_log)

    iodata = load_one(fn_in)

    logger.info("*" * width)
    logger.info(" Molecualr information ".center(width, " "))
    logger.info("*" * width)
    logger.info(f"Atomic numbers : {' '.join([str(z) for z in iodata.atnums])}")
    logger.info("Coordinates [a.u.] : ")
    logger.info(f"{'X':>20} {'Y':>20} {'Z':>20}".center(width, " "))
    for xyz in iodata.atcoords:
        logger.info(f"{xyz[0]:>20.3f} {xyz[1]:>20.3f} {xyz[2]:>20.3f}".center(width, " "))
    logger.info("-" * width)
    logger.info(" ")

    # grid and density
    logger.info(" " * width)
    logger.info("*" * width)
    logger.info(" Build grid and compute moleuclar density ".center(width, " "))
    logger.info("*" * width)

    grid, data = prepare_input(
        iodata,
        args.nrad,
        args.nang,
        args.chunk_size,
        args.gradient,
        args.orbitals,
        True,
    )
    data.update(
        {
            "atcoords": iodata.atcoords,
            "atnums": iodata.atnums,
            "atcorenums": iodata.atcorenums,
            "points": grid.points,
            "weights": grid.weights,
            "aim_weights": grid.aim_weights,
            "cellvecs": np.zeros((0, 3)),
            "nelec": iodata.mo.nelec,
        }
    )

    data["atom_idxs"] = grid._indices
    for iatom in range(iodata.natom):
        atgrid = grid.get_atomic_grid(iatom)
        data[f"atom{iatom}/points"] = atgrid.points
        data[f"atom{iatom}/weights"] = atgrid.weights
        data[f"atom{iatom}/shell_idxs"] = atgrid._indices
        data[f"atom{iatom}/rgrid/points"] = atgrid.rgrid.points
        data[f"atom{iatom}/rgrid/weights"] = atgrid.rgrid.weights

    logger.info("-" * width)
    logger.info(" " * width)
    path = os.path.dirname(fn_out)
    if not os.path.exists(path):
        os.makedirs(path)
    np.savez_compressed(fn_out, **data)
    return 0


def build_parser():
    """Parse command-line arguments."""
    description = """Generate molecular density with HORTON3."""
    parser = argparse.ArgumentParser(prog="part-gen", description=description)

    parser.add_argument(
        "-f",
        "--inputs",
        type=str,
        nargs="+",
        default=None,
        help="The outputs file from quantum chemistry package, e.g., the checkpoint file from Gaussian program.",
    )
    parser.add_argument(
        "--outputs",
        help="The NPZ file in which the grid and the density will be stored.",
        nargs="+",
        type=str,
        default="mol_density.npz",
    )
    # parser.add_argument(
    #     "--verbose",
    #     type=int,
    #     default=3,
    #     help="The level for printing outputs information. [default=%(default)s]",
    # )
    parser.add_argument(
        "--log_level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: %(default)s)",
    )
    parser.add_argument(
        "--log_files",
        type=str,
        nargs="+",
        default=None,
        help="The log file.",
    )
    # for grid
    parser.add_argument(
        "-r",
        "--nrad",
        type=int,
        default=150,
        help="Number of radial grid points. [default=%(default)s]",
    )
    parser.add_argument(
        "-a",
        "--nang",
        type=int,
        default=194,
        help="Number of angular grid points. [default=%(default)s]",
    )
    parser.add_argument(
        "-c",
        "--chunk-size",
        type=int,
        default=10000,
        help="Number points on which the density is computed in one pass. " "[default=%(default)s]",
    )
    # parser.add_argument(
    #     "-s",
    #     "--store-atomic-grids",
    #     default=True,
    #     action="store_true",
    #     dest="store_atgrids",
    #     help="Store atomic integration grids, which may be useful for post-processing. ",
    # )
    # for gbasis
    parser.add_argument(
        "-g",
        "--gradient",
        default=False,
        action="store_true",
        help="Also compute the gradient of the density (and the orbitals). ",
    )
    parser.add_argument(
        "-o",
        "--orbitals",
        default=False,
        action="store_true",
        help="Also store the occupied and virtual orbtials. "
        "For this to work, orbitals must be defined in the WFN file.",
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
