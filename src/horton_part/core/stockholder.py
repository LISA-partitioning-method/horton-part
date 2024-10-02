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
"""Base classes for all stockholder partitioning schemes"""


import time
import warnings

import numpy as np
from grid import LEBEDEV_DEGREES
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from .base import WPart
from .cache import just_once

__all__ = ["AbstractStockholderWPart"]


class AbstractStockholderWPart(WPart):
    """Abstract Stockholder partitioning class."""

    def _init_subgrids(self):
        WPart._init_subgrids(self)
        self.initial_local_grids()

    @property
    def radius_cutoff(self):
        """
        Get the radius of the local grid sphere.

        This property returns the radius of the sphere within which local grids are considered.
        The local grid radius is used in [global methods]. It's a key parameter in [some process].

        Returns
        -------
        float
            The radius of the local grid sphere.

        """
        return np.inf

    @just_once
    def initial_local_grids(self):
        """Compute local grids for each atom.

        This method initializes local grid properties and calculates grids for each atom based on their coordinates and a specified radius.
        """
        self._initialize_grid_properties()
        # TODO: the radius should be basis function dependent.
        for index in range(self.natom):
            self._compute_atom_grid(index)
        self._log_grid_info()

    def _initialize_grid_properties(self):
        """Initialize grid-related properties."""
        # Stores local grids for each atom
        self.local_grids = []
        # Stores overlap of atom and local grid points
        self.atom_points_overlap = []
        # Stores point indices relative to atom grid
        self.pt_indices_relative_to_atom = []
        # Stores radial distances for points in the local grid
        self.radial_distances = []

    def _compute_atom_grid(self, index):
        """Compute grid properties for a specific atom."""
        # Extract local grid around the atom within a user-defined radius
        local_grid = self.grid.get_localgrid(
            center=self.coordinates[index], radius=self.radius_cutoff
        )
        self.local_grids.append(local_grid)

        # Determine indices for the start and end of the current atom's grid
        atom_start_idx, atom_end_idx = (
            self.grid.indices[index],
            self.grid.indices[index + 1],
        )

        # Identify which points in the local grid belong to the current atom
        # Result is a boolean array marking points belonging to the atom
        atom_grid_points = (atom_start_idx <= local_grid.indices) & (
            local_grid.indices < atom_end_idx
        )
        self.atom_points_overlap.append(atom_grid_points)

        # Adjust indices of atom's points to be relative to the atom's grid start index
        # Useful for mapping local grid points to atom grid points
        relative_indices = local_grid.indices[atom_grid_points] - atom_start_idx
        self.pt_indices_relative_to_atom.append(relative_indices)

        # Calculate radial distances from each point in the local grid to the atom's center
        radial_distances = np.linalg.norm(local_grid.points - self.coordinates[index], axis=1)
        self.radial_distances.append(radial_distances)

    def _log_grid_info(self):
        """Log information about the computed grids."""
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("Information of integral grids.")
        self.logger.info("-" * 80)
        self.logger.info("Compute local grids ...")
        self.logger.info(f"Grid size of molecular grid: {self.grid.size}")
        reduced_mol_grid_size = 0
        for i, local_grid in enumerate(self.local_grids):
            self.logger.info(f" Atom {i} ".center(80, "*"))
            atom_grid = self.get_grid(i)
            dist = np.sqrt(np.einsum("ij->i", (atom_grid.points - self.coordinates[i]) ** 2))
            self.logger.info(f"|-- Local grid size: {local_grid.size}")
            reduced_atom_grid_size = len(atom_grid.points[dist <= self.radius_cutoff])
            self.logger.info(f"|-- Atom grid size: {reduced_atom_grid_size}")
            reduced_mol_grid_size += reduced_atom_grid_size
            rgrid = self.get_rgrid(i)
            r = rgrid.points
            if len(r) == len(atom_grid.degrees):
                r_mask = r <= self.radius_cutoff
                degrees = np.asarray(atom_grid.degrees)[r_mask]
                degrees_str = [str(LEBEDEV_DEGREES[d]) for d in degrees]
                self.logger.info(f"   |-- Radial grid size: {len(r[r_mask])}")
                self.logger.info("   |-- Angular grid sizes: ")
                nb = 20
                prefix = "          "
                for j in range(len(degrees_str) // nb + 1):
                    self.logger.info(prefix + " ".join(degrees_str[j * nb : j * nb + nb]))
            else:
                warnings.warn(
                    "The size of 'rgrid' in the method is not equal to the size of 'rgrid' of atom grid."
                )
        self.logger.info("-" * 80)
        self.logger.info(f"Grid size of truncated molecular grid: {reduced_mol_grid_size}")
        self.logger.info("=" * 80)
        self.logger.info(" ")

    def _compute_entropy(self, rho, rho0, density_cutoff=1e-15):
        sick = (rho0 < density_cutoff) | (rho < density_cutoff)
        with np.errstate(all="ignore"):
            ratio = np.divide(rho, rho0, out=np.zeros_like(rho), where=~sick)
            ln_ratio = np.log(ratio, out=np.zeros_like(rho), where=~sick)
        entropy = self._grid.integrate(rho, ln_ratio)
        return entropy

    def update_pro(self, index, proatdens, promoldens):
        """
        Update the pro-molecule density arrays based on the pro-atom density for a specified atom index.

        This method evaluates the pro-atom density for the specified atom index over either a local grid
        (if defined) or the global grid. The evaluated density is then used to update the pro-molecule
        density array. This process contributes to constructing the complete pro-molecule density profile
        by accumulating contributions from individual atoms.

        Parameters
        ----------
        index : int
            The index of the atom for which the pro-atom and pro-molecule densities are to be updated.
        proatdens : 1D np.ndarray
            The array representing the pro-atom density. This array is updated with the new density values
            for the specified atom.
        promoldens : 1D np.ndarray
            The array representing the pro-molecule density. This array accumulates the density contributions
            from each atom, including the one specified by `index`.

        Notes
        -----
        The method checks if `local_grids` attribute is available. If so, it uses the local grid specific
        to the atom index for density evaluation. Otherwise, it defaults to using the global grid.

        The `eval_proatom` method is used to evaluate the pro-atom density, which is then used to update
        the `proatdens` and `promoldens` arrays. A small constant (1e-100) is added to `promoldens` to avoid
        zero values, especially important for iterative methods requiring non-zero initial values.

        If `local_grids` is not present, the method employs the global grid for density evaluation and updates
        the `proatdens` array using the `to_atomic_grid` method.

        """
        if hasattr(self, "local_grids"):
            # TODO: maybe remove the local grids, it is too difficult to debug.
            local_grid = self.local_grids[index]
            work = np.zeros((local_grid.size,))
            self.eval_proatom(index, work, local_grid)
            promoldens[local_grid.indices] += work
            promoldens += 1e-100
            proatdens[self.pt_indices_relative_to_atom[index]] = work[
                self.atom_points_overlap[index]
            ]
        else:
            # work = self.grid.zeros()
            work = np.zeros((self.grid.size,))
            self.eval_proatom(index, work, self.grid)
            promoldens += work
            promoldens += 1e-100
            proatdens[:] = self.to_atomic_grid(index, work)

    def get_rgrid(self, index):
        """Load radial grid.

        Parameters
        ----------
        index: int
            The atom index.
        """
        raise NotImplementedError

    def get_proatom_rho(self, index, *args, **kwargs):
        """Get pro-atom density for atom `iatom`.

        Parameters
        ----------
        index : int
            The atom index
        *args :
            Variable length argument list, used for passing non-keyworded arguments.
        **kwargs :
            Arbitrary keyword arguments, used for passing additional data.

        """
        raise NotImplementedError

    def fix_proatom_rho(self, index, rho, deriv):
        """Check if the radial density for the proatom is correct and fix as needed.

        Parameters
        ----------
        index: int
             The atom for which this proatom rho is created.
        rho: 1D np.ndarray
             The radial density
        deriv: int
             the derivative of the radial density or None.
        """
        rgrid = self.get_rgrid(index)

        # Check for negative parts
        original = rgrid.integrate(rho)
        if rho.min() < 0:
            rho[rho < 0] = 0.0
            deriv = None
            error = rgrid.integrate(rho) - original
            self.logger.info(
                "                Pro-atom not positive everywhere. Lost %.1e electrons" % error
            )
        return rho, deriv

    def get_proatom_spline(self, index, *args, **kwargs):
        """
        Create and return a spline representation of the radial density for a given atomic index.

        This method first retrieves the radial density and its derivatives for the specified atomic index.
        It then ensures the correctness of these values and constructs a spline representation based
        on the radial grid points.

        Parameters
        ----------
        index : int
            The index of the atom for which the radial density spline is to be calculated.
        *args :
            Variable length argument list, used for passing non-keyworded arguments.
        **kwargs :
            Arbitrary keyword arguments, used for passing additional data.

        Returns
        -------
        CubicSpline or CubicHermiteSpline
            A spline representation of the radial density. If derivatives are available,
            a `CubicHermiteSpline` is returned. Otherwise, a `CubicSpline` is used.

        Notes
        -----
        The method internally calls `get_proatom_rho` to obtain the radial density (`rho`) and its
        derivatives (`deriv`), and `fix_proatom_rho` to validate and potentially correct these values.
        It also uses `get_rgrid` to acquire the radial grid points (`rgrid.points`). The spline is
        constructed with these grid points and density values, with the type of spline depending on
        the availability of derivative information.

        """
        # Get the radial density
        rho, deriv = self.get_proatom_rho(index, *args, **kwargs)
        # Double check and fix if needed
        rho, deriv = self.fix_proatom_rho(index, rho, deriv)

        # Make a spline
        rgrid = self.get_rgrid(index)
        if deriv is None:
            return CubicSpline(rgrid.points, rho, True)
        else:
            return CubicHermiteSpline(rgrid.points, rho, deriv, True)

    def eval_spline(self, index, spline, output, grid, label="noname"):
        """
        Evaluate a given spline at radial distances from a specified atom center and store the results in the output array.

        This method calculates the radial distances from the specified atom center to each point in the provided grid.
        It then evaluates the provided spline function at these distances, storing the results in the given output array.

        Parameters
        ----------
        index : int
            The index of the atom whose center is used for calculating radial distances.
        spline : callable
            The spline function to be evaluated. This should be a function that takes an array of radial distances
            and returns the corresponding spline values.
        output : 1D ndarray
            The array where the evaluated spline values will be stored. This array is modified in-place.
        grid : Grid
            An object representing the grid points. It should have an attribute `points` which is an array of
            grid point coordinates.
        label : str, optional
            A label for identification purposes, defaults to "noname".

        Notes
        -----
        The method computes the Euclidean norm (radial distance) from the atom center, specified by `index`, to each
        point in the grid. The spline function is then evaluated at these distances. The results are stored directly
        in the `output` array, overwriting any existing data.

        """
        center = self.coordinates[index]
        r = np.linalg.norm(grid.points - center, axis=1)
        output[:] = spline(r)

    def eval_proatom(self, index, output, grid):
        """
        Evaluate the radial density of a proatom on a given grid and store the results in the output array.

        This method computes the radial density for a specified atomic index by using a spline representation
        of the radial density. The computed values are then stored in the provided output array. A small
        constant is added to the output to avoid zero values, which is crucial for certain methods that
        require non-zero initial values.

        Parameters
        ----------
        index : int
            The index of the atom for which the radial density is to be evaluated.
        output : 1D ndarray
            The array where the evaluated radial density values will be stored. This array is modified in-place.
        grid : Grid
            The grid points where the radial density is to be evaluated.

        Notes
        -----
        The method begins by obtaining a spline representation of the radial density using `get_proatom_spline`.
        It then uses `eval_spline` to evaluate this spline over the provided grid, storing the results in the
        `output` array. The output is then modified by adding a small constant (1e-100) to each element to ensure
        non-zero values, which is crucial for certain iterative self-consistent field methods like ISA.

        A check is performed using `np.isfinite` to ensure that all values in the output array are finite.

        Warnings
        --------
        The addition of a small constant to the output is a workaround for certain limitations in ISA methods
        and might need to be adjusted based on specific use cases.

        See Also
        --------
        grid.Grid

        """
        spline = self.get_proatom_spline(index)
        output[:] = 0.0
        self.eval_spline(index, spline, output, grid, label="proatom")
        # Note: this is very important because the initial values for ISA methods are all zeros.
        # Without this line, the result is wrong.
        # TODO: this should be fixed.
        output += 1e-100
        assert np.isfinite(output).all()

    def update_at_weights(self):
        """See ``Part.update_at_weights``."""
        # This will reconstruct the promolecular density and atomic weights
        # based on the current proatomic splines.
        promoldens = self.cache.load("promoldens", alloc=self.grid.size)[0]
        promoldens[:] = 0

        # update the promolecule density and store the proatoms in the at_weights
        # arrays for later.
        t0 = time.time()
        for index in range(self.natom):
            atmgrid = self.get_grid(index)
            at_weights = self.cache.load("at_weights", index, alloc=atmgrid.size)[0]
            # Here the proatom density is stored in at_weights.
            self.update_pro(index, at_weights, promoldens)
        t1 = time.time()

        # Compute the atomic weights by taking the ratios between proatoms and
        # promolecules.
        for index in range(self.natom):
            # Here, at_weights is proatom density.
            at_weights = self.cache.load("at_weights", index)
            at_weights /= self.to_atomic_grid(index, promoldens)
            np.clip(at_weights, 0, 1, out=at_weights)
            # self.logger.debug("weights:")
            # self.logger.debug(at_weights)
            # self.logger.debug("Negative weights:")
            # self.logger.debug(at_weights[at_weights<-1e-5])
            # self.logger.debug("Weights > 1:")
            # self.logger.debug(at_weights[at_weights>1+1e-5])
        t2 = time.time()

        if "history_time_update_promolecule" not in self.time_usage:
            self.time_usage["history_time_update_promolecule"] = []
            self.time_usage["history_time_compute_at_weights"] = []
        self.time_usage["history_time_update_promolecule"].append(t1 - t0)
        self.time_usage["history_time_compute_at_weights"].append(t2 - t1)

    def do_prosplines(self):
        """Do pro-atom splines"""
        for index in range(self.natom):
            # density
            key = ("spline_prodensity", index)
            if key not in self.cache:
                self.logger.info("Storing proatom density spline for atom %i." % index)
                spline = self.get_proatom_spline(index)
                self.cache.dump(key, spline, tags="o")
            # # hartree potential
            # key = ("spline_prohartree", index)
            # if key not in self.cache:
            #     print(
            #         "Computing proatom hartree potential spline for atom %i." % index
            #     )
            #     rho_spline = self.cache.load("spline_prodensity", index)
            #     v_spline = solve_poisson_becke([rho_spline])[0]
            #     self.cache.dump(key, v_spline, tags="o")
