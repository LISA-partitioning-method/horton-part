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
"""Iterative Stockholder Analysis (ISA) partitioning"""


from __future__ import print_function

import numpy as np

from .cache import just_once
from .stockholder import StockholderWPart
from .log import log
import time


__all__ = ["ISAWPart"]


class ISAWPart(StockholderWPart):
    def __init__(
        self,
        coordinates,
        numbers,
        pseudo_numbers,
        grid,
        moldens,
        spindens=None,
        local=True,
        lmax=3,
        threshold=1e-6,
        maxiter=500,
        inner_threshold=1e-8,
        local_grid_radius=8.0,
    ):
        """
        **Optional arguments:** (that are not defined in ``WPart``)

        threshold
             The procedure is considered to be converged when the maximum
             change of the charges between two iterations drops below this
             threshold.

        maxiter
             The maximum number of iterations. If no convergence is reached
             in the end, no warning is given.
             Reduce the CPU cost at the expense of more memory consumption.

        inner_threshold
            The threshold for inner local optimization problem.

        local_grid_radius
            The radius of the sphere where the local grids are considered.

        """
        self._threshold = threshold
        self._inner_threshold = (
            inner_threshold if inner_threshold < threshold else threshold
        )
        self._maxiter = maxiter
        self._local_grid_radius = local_grid_radius
        StockholderWPart.__init__(
            self,
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            local,
            lmax,
        )

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

        Raises
        ------
        ValueError
            If the local grid radius is not set or out of an expected range.
        """
        if self._local_grid_radius is None or self._local_grid_radius < 0:
            raise ValueError("Local grid radius is not properly set.")
        return self._local_grid_radius

    def compute_change(self, propars1, propars2):
        """Compute the difference between an old and a new proatoms"""
        # Compute mean-square deviation
        msd = 0.0
        for index in range(self.natom):
            rgrid = self.get_rgrid(index)
            rho1, deriv1 = self.get_proatom_rho(index, propars1)
            rho2, deriv2 = self.get_proatom_rho(index, propars2)
            delta = rho1 - rho2
            msd += rgrid.integrate(4 * np.pi * rgrid.points**2, delta, delta)
        return np.sqrt(msd)

    def _init_propars(self):
        """Initial pro-atom parameters and cache lists."""
        self.history_propars = []
        self.history_charges = []
        self.history_entropies = []
        self.history_time_update_at_weights = []
        self.history_time_update_propars_atoms = []

    def _update_propars(self):
        """Update pro-atom parameters."""
        # Keep track of history
        self.history_propars.append(self.cache.load("propars").copy())
        if "promoldens" in self.cache:
            rho = self._moldens
            rho_0 = self.cache.load("promoldens")
            # This is okay, because rho0 and rho are non-negative.
            rho_0 = np.clip(rho_0, 1e-100, np.inf)
            rho = np.clip(rho, 1e-100, np.inf)
            entropy = self._grid.integrate(rho, np.log(rho) - np.log(rho_0))
            if log.do_medium:
                print(f"Entropy: {entropy:.8f}")
            self.history_entropies.append(entropy)

        # Update the partitioning based on the latest proatoms
        t0 = time.time()
        self.update_at_weights()
        t1 = time.time()

        # Update the proatoms
        for index in range(self.natom):
            self._update_propars_atom(index)
        t2 = time.time()

        self.history_time_update_at_weights.append(t1 - t0)
        self.history_time_update_propars_atoms.append(t2 - t1)

        # Keep track of history
        self.history_charges.append(self.cache.load("charges").copy())

    def _update_propars_atom(self, index):
        """Update pro-atom parameters for each atom."""
        raise NotImplementedError

    def _finalize_propars(self):
        """Restore the pro-atom parameters."""
        charges = self._cache.load("charges")
        self.cache.dump("history_propars", np.array(self.history_propars), tags="o")
        self.cache.dump("history_charges", np.array(self.history_charges), tags="o")
        self.cache.dump("history_entropies", np.array(self.history_entropies), tags="o")
        self.cache.dump("populations", self.numbers - charges, tags="o")
        self.cache.dump("pseudo_populations", self.pseudo_numbers - charges, tags="o")
        self.cache.dump(
            "history_time_update_at_weights",
            np.array(self.history_time_update_at_weights),
            tags="o",
        )
        self.cache.dump(
            "history_time_update_propars_atoms",
            np.array(self.history_time_update_propars_atoms),
            tags="o",
        )
        self.cache.dump(
            "time_update_at_weights",
            np.sum(self.history_time_update_at_weights),
            tags="o",
        )
        self.cache.dump(
            "time_update_propars_atoms",
            np.sum(self.history_time_update_propars_atoms),
            tags="o",
        )
        self.cache.dump(
            "history_time_update_promolecule",
            np.array(self.time_usage["history_time_update_promolecule"]),
            tags="o",
        )
        self.cache.dump(
            "history_time_compute_at_weights",
            np.array(self.time_usage["history_time_compute_at_weights"]),
            tags="o",
        )
        self.cache.dump(
            "time_update_promolecule",
            np.sum(self.time_usage["history_time_update_promolecule"]),
            tags="o",
        )
        self.cache.dump(
            "time_compute_at_weights",
            np.sum(self.time_usage["history_time_compute_at_weights"]),
            tags="o",
        )

    @just_once
    def do_partitioning(self):
        """Do partitioning"""
        self.initial_local_grids()
        # Perform one general check in the beginning to avoid recomputation
        new = any(("at_weights", i) not in self.cache for i in range(self.natom))
        new |= "niter" not in self.cache
        new |= "change" not in self.cache
        if new:
            propars = self._init_propars()
            print("Iteration       Change")

            counter = 0
            while True:
                counter += 1
                self.cache.dump("niter", counter, tags="o")

                # Update the parameters that determine the pro-atoms.
                old_propars = propars.copy()
                self._update_propars()

                # Check for convergence
                change = self.compute_change(propars, old_propars)
                print("%9i   %10.5e" % (counter, change))
                if change < self._threshold or counter >= self._maxiter:
                    break
            print()

            self._finalize_propars()
            self.cache.dump("niter", counter, tags="o")
            self.cache.dump("change", change, tags="o")
