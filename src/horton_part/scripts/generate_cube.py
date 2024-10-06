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

import numpy as np
from grid import UniformGrid
from iodata import load_one

from horton_part.core.basis import BasisFuncHelper
from horton_part.scripts.generate_density import _compute_stuff
from horton_part.scripts.program import PartProg

np.set_printoptions(precision=14, suppress=True, linewidth=np.inf)
np.random.seed(44)

__all__ = ["prepare_input_cube", "PartCubeProg", "_setup_cube_grid"]


def prepare_input_cube(iodata, chunk_size, gradient, orbitals, logger, grid=None):
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
    if grid is None:
        grid = _setup_cube_grid(iodata.atnums, iodata.atcoords, logger)
    data = _compute_stuff(iodata, grid.points, gradient, orbitals, chunk_size, logger)
    return grid, data


# pylint: disable=protected-access
def _setup_cube_grid(atnums, atcoords, logger, spacing=0.2, extension=5.0):
    """Set up a simple molecular integration grid for a given molecular geometry.

    Parameters
    ----------
    atnums: np.ndarray(N,)
        Atomic numbers
    atcoords: np.ndarray(N, 3)
        Atomic coordinates.

    Returns
    -------
    grid
        A molecular integration grid, instance (of a subclass of)
        grid.basegrid.Grid.

    """
    logger.info("Setting up grid")
    grid = UniformGrid.from_molecule(
        atnums, atcoords, rotate=False, spacing=spacing, extension=extension
    )
    assert np.isfinite(grid.points).all()
    assert np.isfinite(grid.weights).all()
    assert (grid.weights >= 0).all()
    # TODO: remove grid points with zero weight
    return grid


def to_cube(fname, atnums, atcorenums, atcoords, grid: UniformGrid, data):
    r"""Write the data evaluated on grid points into a cube file.

    Parameters
    ----------
    fname : str
        Cube file name with \*.cube extension.
    data : np.ndarray, shape=(npoints,)
        An array containing the evaluated scalar property on the grid points.
    """
    if not fname.endswith(".cube"):
        raise ValueError("Argument fname should be a cube file with `*.cube` extension!")
    if data.size != grid.size:
        raise ValueError(
            "Argument data should have the same size as the grid. "
            + f"{data.size}!={grid._npoints}"
        )

    # Write data into the cube file
    with open(fname, "w") as f:
        # writing the cube header:
        f.write("Cubefile created with HORTON-PART\n")
        f.write("OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
        natom = len(atnums)
        x, y, z = grid.origin
        f.write(f"{natom:5d} {x:11.6f} {y:11.6f} {z:11.6f}\n")
        rvecs = grid.axes
        for i, (x, y, z) in zip(grid.shape, rvecs):
            f.write(f"{i:5d} {x:11.6f} {y:11.6f} {z:11.6f}\n")
        for i, q, (x, y, z) in zip(atnums, atcorenums, atcoords):
            f.write(f"{i:5d} {q:11.6f} {x:11.6f} {y:11.6f} {z:11.6f}\n")
        # writing the cube data:
        num_chunks = 6
        for i in range(0, data.size, num_chunks):
            row_data = data.flat[i : i + num_chunks]
            f.write((row_data.size * " {:12.5E}").format(*row_data))
            f.write("\n")


def _compute_rho0(atnums, points, pops, func_type="gauss", nderiv=0):
    if func_type in ["gauss", "slater"]:
        bs_helper = BasisFuncHelper.from_function_type(func_type)
    elif ".json" in func_type and os.path.exists(func_type):
        bs_helper = BasisFuncHelper.from_json(func_type)
    else:
        raise RuntimeError(f"Invalid func_type: {func_type}!")

    begin = 0
    rho0 = np.zeros_like(points)
    for i, number in enumerate(atnums):
        nshell = bs_helper.get_nshell(number)
        rho0[i, :] = bs_helper.compute_proatom_dens(
            number, pops[begin : begin + nshell], points[i, :], nderiv
        )
        begin += nshell
    return rho0


class PartCubeProg(PartProg):
    """Part-Cube Program"""

    def __init__(self, width=100):
        description = """Generate cube file for a molecular density."""
        super().__init__("part-cube", width, description=description)

    def single_launch(self, settings, fn_in, fn_out, fn_log, **kwargs):
        ifile = kwargs["ifile"]
        self.setup_logger(settings, fn_log)
        self.print_settings(settings, fn_in, fn_out, fn_log)
        iodata = load_one(fn_in)
        self.print_header("Molecular information")
        self.print_coordinates(iodata.atnums, iodata.atcoords)
        self.print_header("Build grid and compute molecular density")

        grid, data = prepare_input_cube(
            iodata,
            settings["chunk_size"],
            settings["gradient"],
            settings["orbitals"],
            self.logger,
        )
        data.update(
            {
                "atcoords": iodata.atcoords,
                "atnums": iodata.atnums,
                "atcorenums": iodata.atcorenums,
                "points": grid.points,
                "weights": grid.weights,
                "origin": grid.origin,
                "shape": grid.shape,
                "axes": grid.axes,
                "nelec": iodata.mo.nelec,
            }
        )

        if settings["with_partdens"]:
            # The filename of output obtained by part-dens program
            fn_partden = settings["partdens"][ifile]
            partden_data = np.load(fn_partden)
            # Load the optimized propars obtained by LISA/gLISA methods
            propars = partden_data["history_propars"][-1, :]
            # The distance array
            dis_array = np.linalg.norm(
                grid.points[None, :, :] - iodata.atcoords[:, None, :], axis=2
            )
            # Compute pro-atom density and store it in an array, 2d array and shape = N_atoms * N_points
            rho0 = _compute_rho0(iodata.atnums, dis_array, propars, settings["basis_func"])
            # Compute pro-molecule density, 1d array and shape = N_points
            promol = np.sum(rho0, axis=0)
            promol += 1e-100
            # Compute AIM weights function, 2d array and shape = N_atoms * N_points
            weights_funcs = rho0 / promol
            density = data["density"]
            # Compute AIM density, 2d array and N_atom * N_points
            aim_rho = weights_funcs * density[None, :]

            self.logger.info(f"The integral of density: {grid.integrate(density):.3f}")
            self.logger.info(
                f"The number of electrons in the molecule: {np.sum(iodata.atcorenums)}"
            )

        if settings["with_partdens"] and settings["with_aim_cache"]:
            data.update(
                {
                    "rho0": rho0,
                    "aim_rho": aim_rho,
                    "density": density,
                }
            )

        path = os.path.dirname(os.path.abspath(fn_out))
        if not os.path.exists(path):
            os.makedirs(path)
        np.savez_compressed(fn_out, **data)

        if settings["with_cube_files"]:
            fn_name = ".".join(fn_out.split(".")[:-1])
            to_cube(
                f"{fn_name}_rho_mol.cube",
                iodata.atnums,
                iodata.atcorenums,
                iodata.atcoords,
                grid,
                data["density"],
            )
            if settings["with_partdens"]:
                for i in range(len(iodata.atnums)):
                    to_cube(
                        f"{fn_name}_rho_{i}.cube",
                        iodata.atnums,
                        iodata.atcorenums,
                        iodata.atcoords,
                        grid,
                        aim_rho[i, :],
                    )
                    to_cube(
                        f"{fn_name}_rho0_{i}.cube",
                        iodata.atnums,
                        iodata.atcorenums,
                        iodata.atcoords,
                        grid,
                        rho0[i, :],
                    )
                to_cube(
                    f"{fn_name}_rho0_mol.cube",
                    iodata.atnums,
                    iodata.atcorenums,
                    iodata.atcoords,
                    grid,
                    promol,
                )
        return 0


def main(args=None) -> int:
    """Main entry."""
    return PartCubeProg().run(args)


if __name__ == "__main__":
    sys.exit(main())
