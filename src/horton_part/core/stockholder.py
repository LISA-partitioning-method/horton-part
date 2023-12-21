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
"""Base classes for all stockholder partitioning schemes"""


import numpy as np
import time
import warnings
import logging

from .base import WPart, just_once

from scipy.interpolate import CubicSpline, CubicHermiteSpline


__all__ = ["AbstractStockholderWPart"]

logger = logging.getLogger(__name__)


def eval_spline_grid(spline, grid, center):
    r = np.linalg.norm(grid.points - center, axis=1)
    return spline(r)


class AbstractStockholderWPart(WPart):
    def _init_subgrids(self):
        WPart._init_subgrids(self)
        self.initial_local_grids()

    @property
    def local_grid_radius(self):
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
            center=self.coordinates[index], radius=self.local_grid_radius
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
        radial_distances = np.linalg.norm(
            local_grid.points - self.coordinates[index], axis=1
        )
        self.radial_distances.append(radial_distances)

    def _log_grid_info(self):
        """Log information about the computed grids."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("Information of integral grids.")
        logger.info("-" * 80)
        logger.info("Compute local grids ...")
        logger.info(f"Grid size of molecular grid: {self.grid.size}")
        reduced_mol_grid_size = 0
        for i, local_grid in enumerate(self.local_grids):
            logger.info(f" Atom {i} ".center(80, "*"))
            atom_grid = self.get_grid(i)
            dist = np.sqrt(
                np.einsum("ij->i", (atom_grid.points - self.coordinates[i]) ** 2)
            )
            logger.info(f"|-- Local grid size: {local_grid.size}")
            reduced_atom_grid_size = len(
                atom_grid.points[dist <= self.local_grid_radius]
            )
            logger.info(f"|-- Atom grid size: {reduced_atom_grid_size}")
            reduced_mol_grid_size += reduced_atom_grid_size
            rgrid = self.get_rgrid(i)
            r = rgrid.points
            if len(r) == len(atom_grid.degrees):
                r_mask = r <= self.local_grid_radius
                degrees = np.asarray(atom_grid.degrees)[r_mask]
                degrees_str = [str(d) for d in degrees]
                logger.info(f"   |-- Radial grid size: {len(r[r_mask])}")
                logger.info(f"   |-- Angular grid {len(degrees)} degrees: ")
                nb = 20
                prefix = "          "
                for j in range(len(degrees_str) // nb + 1):
                    logger.info(prefix + " ".join(degrees_str[j * nb : j * nb + nb]))
            else:
                warnings.warn(
                    "The size of 'rgrid' in the method is not equal to the size of 'rgrid' of atom grid."
                )
        logger.info("-" * 80)
        logger.info(f"Grid size of truncated molecular grid: {reduced_mol_grid_size}")
        logger.info("=" * 80)
        logger.info(" ")

    def update_pro(self, index, proatdens, promoldens):
        if hasattr(self, "local_grids"):
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
            proatdens[:] = self.to_atomic_grid(index, work)

    def get_rgrid(self, index):
        """Load radial grid."""
        raise NotImplementedError

    def get_proatom_rho(self, index, *args, **kwargs):
        """Get pro-atom density for atom `iatom`."""
        raise NotImplementedError

    def fix_proatom_rho(self, index, rho, deriv):
        """Check if the radial density for the proatom is correct and fix as needed.

        **Arguments:**

        index
             The atom for which this proatom rho is created.

        rho
             The radial density

        deriv
             the derivative of the radial density or None.
        """
        rgrid = self.get_rgrid(index)

        # Check for negative parts
        original = rgrid.integrate(rho)
        if rho.min() < 0:
            rho[rho < 0] = 0.0
            deriv = None
            error = rgrid.integrate(rho) - original
            logger.info(
                "                Pro-atom not positive everywhere. Lost %.1e electrons"
                % error
            )
        return rho, deriv

    def get_proatom_spline(self, index, *args, **kwargs):
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
        center = self.coordinates[index]
        output[:] = eval_spline_grid(spline, grid, center)

    def eval_proatom(self, index, output, grid):
        spline = self.get_proatom_spline(index)
        output[:] = 0.0
        self.eval_spline(index, spline, output, grid, label="proatom")
        # Note: this is very important because the initial values for ISA methods are all zeros.
        # Without this line, the result is wrong.
        # TODO: this should be fixed.
        output += 1e-100
        assert np.isfinite(output).all()

    def update_at_weights(self):
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
            self.update_pro(index, at_weights, promoldens)
        t1 = time.time()

        # Compute the atomic weights by taking the ratios between proatoms and
        # promolecules.
        for index in range(self.natom):
            at_weights = self.cache.load("at_weights", index)
            at_weights /= self.to_atomic_grid(index, promoldens)
            np.clip(at_weights, 0, 1, out=at_weights)
            # logger.debug("weights:")
            # logger.debug(at_weights)
            # logger.debug("Negative weights:")
            # logger.debug(at_weights[at_weights<-1e-5])
            # logger.debug("Weights > 1:")
            # logger.debug(at_weights[at_weights>1+1e-5])
        t2 = time.time()

        if "history_time_update_promolecule" not in self.time_usage:
            self.time_usage["history_time_update_promolecule"] = []
            self.time_usage["history_time_compute_at_weights"] = []
        self.time_usage["history_time_update_promolecule"].append(t1 - t0)
        self.time_usage["history_time_compute_at_weights"].append(t2 - t1)

    def do_prosplines(self):
        for index in range(self.natom):
            # density
            key = ("spline_prodensity", index)
            if key not in self.cache:
                logger.info("Storing proatom density spline for atom %i." % index)
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
