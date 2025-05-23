# HORTON-PART: molecular density partition schemes based on HORTON package.
# Copyright (C) 2023-2025 The HORTON-PART Development Team
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

import numpy as np
from gbasis.evals.eval import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis
from gbasis.wrappers import from_iodata
from grid.becke import BeckeWeights
from grid.molgrid import MolGrid
from grid.onedgrid import GaussChebyshev
from grid.rtransform import BeckeRTransform
from iodata import load_one

from horton_part.scripts.program import PartProg

np.set_printoptions(precision=14, suppress=True, linewidth=np.inf)
np.random.seed(44)

__all__ = ["prepare_input", "PartGenProg"]

"""
This file contains code derived from the `Denspart` package, which is designed for generating molecular densities
using Horton3 interface. For more details about the `Denspart` package, including its features and API, see:
https://github.com/theochem/denspart.
"""


def prepare_input(iodata, nrad, nang, chunk_size, gradient, orbitals, store_atgrids, logger):
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
    grid = _setup_grid(iodata.atnums, iodata.atcoords, nrad, nang, store_atgrids, logger)
    data = _compute_stuff(iodata, grid.points, gradient, orbitals, chunk_size, logger)
    return grid, data


# pylint: disable=protected-access
def _setup_grid(atnums, atcoords, nrad, nang, store_atgrids, logger):
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
    grid = MolGrid.from_size(atnums, atcoords, nang, rgrid, becke, store=store_atgrids, rotate=0)
    assert np.isfinite(grid.points).all()
    assert np.isfinite(grid.weights).all()
    assert (grid.weights >= 0).all()
    # TODO: remove grid points with zero weight
    return grid


def _compute_stuff(iodata, points, gradient, orbitals, chunk_size, logger):
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
    # basis, coord_types = from_iodata(iodata)
    basis = from_iodata(iodata)

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
        basis_grid = evaluate_basis(basis, points[istart:iend])
        if gradient:
            logger.info("  basis_gradient")
            basis_gradient_grid = np.array(
                [
                    evaluate_deriv_basis(basis, points[istart:iend], orders)
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


class PartGenProg(PartProg):
    """Part-Gen Program"""

    def __init__(self, width=100):
        description = """Generate molecular density with HORTON3."""
        super().__init__("part-gen", width, description=description)

    def single_launch(self, settings, fn_in, fn_out, fn_log, **kwargs):
        self.setup_logger(settings, fn_log)
        self.print_settings(settings, fn_in, fn_out, fn_log)
        iodata = load_one(fn_in)
        self.print_header("Molecular information")
        self.print_coordinates(iodata.atnums, iodata.atcoords)
        self.print_header("Build grid and compute molecular density")

        grid, data = prepare_input(
            iodata,
            settings["nrad"],
            settings["nang"],
            settings["chunk_size"],
            settings["gradient"],
            settings["orbitals"],
            True,
            self.logger,
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

        path = os.path.dirname(os.path.abspath(fn_out))
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez_compressed(fn_out, **data)
        return 0


def main(args=None) -> int:
    """Main entry."""
    return PartGenProg().run(args)


if __name__ == "__main__":
    sys.exit(main())
