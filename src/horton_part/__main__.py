# -*- coding: utf-8 -*-
# HORTON-PART: GRID for Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2023 The HORTON-PART Development Team
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
"""Prepare inputs for denspart with HORTON3 modules.

This implementation makes some ad hoc choices on the molecular integration
grid, which should be revised in future. For now, this is something that is
just supposed to just work. The code is not tuned for precision nor for
efficiency.

This module is far from polished and is currently only used for prototyping:

- The integration grid is not final. It may have precision issues and it is not
  pruned. Furthermore, all atoms are given the same atomic grid, which is far
  from optimal and the Becke partitioning is used, which can be improved upon.

- Only the spin-summed density is computed, using the post-hf 1RDM if it is
  present. The spin-difference density is ignored.

- It is slow.

"""


import argparse

import numpy as np
from gbasis.evals.eval import evaluate_basis
from gbasis.evals.eval_deriv import evaluate_deriv_basis
from gbasis.wrappers import from_iodata
from grid.becke import BeckeWeights
from grid.molgrid import MolGrid
from grid.onedgrid import GaussChebyshev
from grid.rtransform import BeckeRTransform
from grid.basegrid import OneDGrid
from grid.atomgrid import AtomGrid
from iodata import load_one
from horton_part import wpart_schemes, log

width = 100
np.set_printoptions(precision=14, suppress=True, linewidth=width)
np.random.seed(44)

__all__ = ["prepare_input", "construct_molgrid_from_dict"]


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
    print("Setting up grid")
    becke = BeckeWeights(order=3)
    # Fix for missing radii.
    becke._radii[2] = 0.5
    becke._radii[10] = 1.0
    becke._radii[18] = 2.0
    becke._radii[36] = 2.5
    becke._radii[54] = 3.5
    oned = GaussChebyshev(nrad)
    rgrid = BeckeRTransform(1e-4, 1.5).transform_1d_grid(oned)
    grid = MolGrid.from_size(
        atnums, atcoords, rgrid, nang, becke, store=store_atgrids, rotate=0
    )
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
                "The input file lacks wavefunction data with which "
                "the density can be computed."
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
        print(f"Computing stuff: {istart} ... {iend} / {len(points)}")
        # Basis functions are computed upfront for efficiency.
        print("  basis")
        basis_grid = evaluate_basis(basis, points[istart:iend], coord_type=coord_types)
        if gradient:
            print("  basis_gradient")
            basis_gradient_grid = np.array(
                [
                    evaluate_deriv_basis(
                        basis, points[istart:iend], orders, coord_type=coord_types
                    )
                    for orders in np.identity(3, dtype=int)
                ]
            )
        # Use basis functions on grid for various quantities.
        print("  density")
        result["density"][istart:iend] = np.einsum(
            "ab,bp,ap->p", one_rdm, basis_grid, basis_grid, optimize=True
        )
        if gradient:
            print("  density gradient")
            result["density_gradient"][istart:iend] = 2 * np.einsum(
                "ab,bp,cap->pc", one_rdm, basis_grid, basis_gradient_grid, optimize=True
            )
        if orbitals:
            print("  orbitals")
            result["orbitals"][istart:iend] = np.einsum(
                "bo,bp->po", iodata.mo.coeffs, basis_grid
            )
            if gradient:
                print("  orbitals gradient")
                result["orbitals_gradient"][istart:iend] = np.einsum(
                    "bo,cbp->poc", iodata.mo.coeffs, basis_gradient_grid
                )
        istart = iend
    assert (result["density"] >= 0).all()
    return result


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


def main(args=None):
    """Command-line interface."""
    args = parse_args(args)
    log.set_level(args.verbose)

    if not args.use_cache:
        print("Loading file : ", args.fn_wfn)
        iodata = load_one(args.fn_wfn)
        print("*" * width)
        print(" Molecualr information ".center(width, " "))
        print("*" * width)
        print("atnums :")
        print(iodata.atnums)
        print("Coordinates [a.u.]: ")
        print(iodata.atcoords)

        # grid and density
        print(" " * width)
        print("*" * width)
        print(" Build grid and compute moleuclar density ".center(width, " "))
        print("*" * width)
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
        print(" " * width)
        print("*" * width)
    else:
        print("*" * width)
        print(f"Reade grid and density data from {args.cache}")
        print("*" * width)
        data = np.load(args.cache)
        grid = construct_molgrid_from_dict(data)

    # partitioning
    if not args.skip_part:
        print(" " * width)
        print("*" * width)
        print(" Partitioning ".center(width, " "))
        print("*" * width)
        kwargs = {
            "coordinates": data["atcoords"],
            "numbers": data["atnums"],
            "pseudo_numbers": data["atcorenums"],
            "grid": grid,
            "moldens": data["density"],
            "lmax": args.lmax,
            "maxiter": args.maxiter,
            "threshold": args.threshold,
            "local_grid_radius": args.local_grid_radius,
        }

        if args.type in ["gisa", "lisa"]:
            kwargs["solver"] = args.solver
            kwargs["basis_func_type"] = args.func_type
            if args.type in ["lisa"]:
                kwargs["use_global_method"] = args.use_global_method
                if args.solver > 200:
                    kwargs["diis_size"] = args.diis_size

        part = wpart_schemes(args.type)(**kwargs)
        part.do_partitioning()
        # part.do_moments()

        print(" " * width)
        print("*" * width)
        print(" Results ".center(width, " "))
        print("*" * width)
        print("charges:")
        print(part.cache["charges"])
        # print("cartesian multipoles:")
        # print(part.cache["cartesian_multipoles"])
        # print("radial moments:")
        # print(part.cache["radial_moments"])

        if not (args.type in ["lisa"] and args.use_global_method):
            print(" " * width)
            print("*" * width)
            print(" Time usage ".center(width, " "))
            print("*" * width)
            print(
                f"Do Partitioning                              : {part.time_usage['do_partitioning']:>10.2f} s"
            )
            print(
                f"  Update Weights                             : {part._cache['time_update_at_weights']:>10.2f} s"
            )
            print(
                f"    Update Promolecule Density (N_atom**2)   : {part._cache['time_update_promolecule']:>10.2f} s"
            )
            print(
                f"    Update AIM Weights (N_atom)              : {part._cache['time_compute_at_weights']:>10.2f} s"
            )
            print(
                f"  Update Atomic Parameters (iter*N_atom)     : {part._cache['time_update_propars_atoms']:>10.2f} s"
            )
            # print(f"Do Moments                                   : {part.time_usage['do_moments']:>10.2f} s")
            print("*" * width)
            print(" " * width)

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

        if not (args.type in ["lisa"] and args.use_global_method):
            part_data["time"] = part.time_usage["do_partitioning"]
            part_data["time_update_at_weights"] = part._cache["time_update_at_weights"]
            part_data["time_update_promolecule"] = part._cache[
                "time_update_promolecule"
            ]
            part_data["time_compute_at_weights"] = part._cache[
                "time_compute_at_weights"
            ]
            part_data["time_update_propars_atoms"] = part._cache[
                "time_update_propars_atoms"
            ]
            part_data["niter"] = part.cache["niter"]
            part_data["history_charges"] = part.cache["history_charges"]
            part_data["history_propars"] = part.cache["history_propars"]
            part_data["history_entropies"] = part.cache["history_entropies"]

        # part_data["part/cartesian_multipoles"] = part.cache["cartesian_multipoles"]
        # part_data["part/radial_moments"] = part.cache["radial_moments"]
        np.savez_compressed(args.output, **part_data)
    else:
        np.savez_compressed(args.output, **data)


def parse_args(args=None):
    """Parse command-line arguments."""
    description = "Molecular density partitioning with HORTON3."
    parser = argparse.ArgumentParser(prog="part", description=description)
    parser.add_argument("--fn_wfn", type=str, help="The wavefunction file.")
    parser.add_argument(
        "--output",
        help="The NPZ file in which the grid and the density will be stored.",
        type=str,
        default="results.npz",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=3,
        help="The level for printing output information. [default=%(default)s]",
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
        help="Number points on which the density is computed in one pass. "
        "[default=%(default)s]",
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
        "--skip_part",
        default=False,
        action="store_true",
        help="Skip partitioning ",
    )
    # for part
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        default="lisa",
        choices=["gisa", "lisa", "mbis", "is"],
        help="Number of angular grid points. [default=%(default)s]",
    )
    parser.add_argument(
        "--func_type",
        type=str,
        default="gauss",
        choices=["gauss", "exp"],
        help="The type of basis functions. [default=%(default)s]",
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
        "--use_cache",
        default=False,
        action="store_true",
        help="Use grid and density from previous calculations",
    )
    parser.add_argument(
        "--cache",
        help="The NPZ file in which the grid and the density will be stored.",
        type=str,
        default="results.npz",
    )
    parser.add_argument(
        "--diis_size",
        type=int,
        default=8,
        help="The number of previous iterations info used in DIIS. [default=%(default)s]",
    )
    parser.add_argument(
        "--use_global_method",
        default=False,
        action="store_true",
        help="Whether use global method",
    )
    parser.add_argument(
        "--local_grid_radius",
        type=float,
        default=np.inf,
        help="The radius for local atomic grid [default=%(default)s]",
    )

    return parser.parse_args(args=args)


if __name__ == "__main__":
    main()
