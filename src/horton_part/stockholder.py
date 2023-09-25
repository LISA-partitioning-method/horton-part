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


from __future__ import print_function

import numpy as np
import time

from .base import WPart, just_once

from scipy.interpolate import CubicSpline, CubicHermiteSpline


__all__ = ["StockholderWPart"]


def eval_spline_grid(spline, grid, center):
    r = np.linalg.norm(grid.points - center, axis=1)
    return spline(r)


class StockholderWPart(WPart):
    @just_once
    def compute_local_grids(self):
        self.local_grids = []
        self.atom_in_local_grid = []
        self.points_in_atom = []
        # TODO: the radius should be basis function dependent.
        for index in range(self.natom):
            local_grid = self.grid.get_localgrid(
                center=self.coordinates[index], radius=8.0
            )
            self.local_grids.append(local_grid)
            begin, end = self.grid.indices[index], self.grid.indices[index + 1]
            in_atm = (begin <= local_grid.indices) & (local_grid.indices < end)
            self.atom_in_local_grid.append(in_atm)
            self.points_in_atom.append(local_grid.indices[in_atm] - begin)

    def update_pro(self, index, proatdens, promoldens):
        # work = np.zeros((self.grid.size,))
        # self.eval_proatom(index, work, self.grid)
        # promoldens += work
        # proatdens[:] = self.to_atomic_grid(index, work)

        self.compute_local_grids()
        local_grid = self.local_grids[index]
        work = np.zeros((local_grid.size,))
        self.eval_proatom(index, work, local_grid)
        promoldens[local_grid.indices] += work
        promoldens += 1e-100
        proatdens[self.points_in_atom[index]] = work[self.atom_in_local_grid[index]]

    def get_rgrid(self, index):
        raise NotImplementedError

    def get_proatom_rho(self, index, *args, **kwargs):
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
            print(
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
                print("Storing proatom density spline for atom %i." % index)
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
