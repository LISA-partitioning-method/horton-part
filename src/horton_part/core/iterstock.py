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
"""Iterative Stockholder Analysis (ISA) partitioning"""

import time

import numpy as np

from .cache import just_once
from .stockholder import AbstractStockholderWPart

__all__ = ["AbstractISAWPart", "compute_change"]


def compute_change(part, propars1, propars2):
    """Compute the difference between an old and a new proatoms"""
    # Compute mean-square deviation
    msd = 0.0
    for index in range(part.natom):
        rho1, deriv1 = part.get_proatom_rho(index, propars1)
        rho2, deriv2 = part.get_proatom_rho(index, propars2)
        delta = rho1 - rho2
        if part.grid_type in [1, 3]:
            rgrid = part.get_rgrid(index)
            msd += rgrid.integrate(4 * np.pi * rgrid.points**2, delta, delta)
        elif part.grid_type == 2:
            grid = part.grid
            msd += grid.integrate(delta, delta)
        else:
            raise NotImplementedError
    return np.sqrt(msd)


def finalize_propars(part):
    """Restore the pro-atom parameters."""
    charges = part._cache.load("charges")
    part.cache.dump("history_propars", np.array(part.history_propars), tags="o")
    part.cache.dump("history_charges", np.array(part.history_charges), tags="o")
    part.cache.dump("history_entropies", np.array(part.history_entropies), tags="o")
    part.cache.dump("history_changes", np.array(part.history_changes), tags="o")
    part.cache.dump("populations", part.numbers - charges, tags="o")
    part.cache.dump("pseudo_populations", part.pseudo_numbers - charges, tags="o")
    part.cache.dump("time_update_at_weights", np.sum(part.history_time_update_at_weights), tags="o")
    part.cache.dump("time_update_propars", np.sum(part.history_time_update_propars), tags="o")


class AbstractISAWPart(AbstractStockholderWPart):
    """Abstract Iterative Stockholder Analysis class."""

    def __init__(
        self,
        coordinates,
        numbers,
        pseudo_numbers,
        grid,
        moldens,
        spindens=None,
        lmax=3,
        logger=None,
        # New parameters added
        threshold=1e-6,
        maxiter=500,
        inner_threshold=1e-8,
        # radius_cutoff=np.inf,
        grid_type=1,
        **kwargs,
    ):
        """
        Initial function.

        **Optional arguments:** (that are not defined in :class:`horton_part.core.base.WPart`)

        Parameters
        ----------
        threshold : float
             The procedure is considered to be converged when the maximum
             change of the charges between two iterations drops below this
             threshold.
        maxiter : int
             The maximum number of iterations. If no convergence is reached
             in the end, no warning is given.
             Reduce the CPU cost at the expense of more memory consumption.
        inner_threshold : float
            The threshold for inner local optimization problem.

        """
        self._threshold = threshold
        self._inner_threshold = inner_threshold if inner_threshold < threshold else threshold
        self._maxiter = maxiter
        # self._radius_cutoff = radius_cutoff
        AbstractStockholderWPart.__init__(
            self,
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            lmax,
            logger,
            grid_type=grid_type,
        )

    # @property
    # def radius_cutoff(self):
    #     """
    #     Get the radius of the local grid sphere.

    #     This property returns the radius of the sphere within which local grids are considered.
    #     The local grid radius is used in [global methods]. It's a key parameter in [some process].

    #     Returns
    #     -------
    #     float
    #         The radius of the local grid sphere.

    #     Raises
    #     ------
    #     ValueError
    #         If the local grid radius is not set or out of an expected range.
    #     """
    #     if self._radius_cutoff is None or self._radius_cutoff < 0:
    #         raise ValueError("Local grid radius is not properly set.")
    #     return self._radius_cutoff

    def compute_change(self, propars1, propars2):
        """Compute the difference between an old and a new proatoms"""
        # Compute mean-square deviation
        return compute_change(self, propars1, propars2)

    def _init_propars(self):
        """Initial pro-atom parameters and cache lists."""
        raise NotImplementedError

    def _update_entropy(self):
        if "promoldens" in self.cache:
            rho = self._moldens
            rho0 = self.cache.load("promoldens")
            entropy = self._compute_entropy(rho, rho0)
            self.history_entropies.append(entropy)

    def _update_propars(self):
        """Update pro-atom parameters."""
        # Update the partitioning based on the latest proatoms
        t0 = time.time()
        self.update_at_weights()
        t1 = time.time()

        # Update the proatoms
        for index in range(self.natom):
            self._update_propars_atom(index)
        t2 = time.time()

        self.history_time_update_at_weights.append(t1 - t0)
        self.history_time_update_propars.append(t2 - t1)

        # Keep track of history
        self.history_propars.append(self.cache.load("propars").copy())
        self.history_charges.append(self.cache.load("charges").copy())

    def _update_propars_atom(self, index):
        """Update pro-atom parameters for each atom."""
        raise NotImplementedError

    def _finalize_propars(self):
        """Restore the pro-atom parameters."""
        finalize_propars(self)

    @just_once
    def do_partitioning(self):
        """Do partitioning"""
        # self.initial_local_grids()
        # Perform one general check in the beginning to avoid recomputation
        new = any(("at_weights", i) not in self.cache for i in range(self.natom))
        new |= "niter" not in self.cache
        new |= "change" not in self.cache
        if new:
            propars = self._init_propars()
            self.logger.info("Iteration       Change      Entropy")
            # self.history_propars.append(propars.copy())

            counter = 0
            while True:
                counter += 1
                self.cache.dump("niter", counter, tags="o")

                # Update the parameters that determine the pro-atoms.
                old_propars = propars.copy()
                self._update_propars()
                # NOTE: the entropy corresponds to old_propars, while the change is current propars - old_propars
                self._update_entropy()

                # Check for convergence
                change = self.compute_change(propars, old_propars)
                entropy = self.history_entropies[counter - 1]
                self.history_changes.append(change)
                self.logger.info("%9i   %10.5e   %10.5e" % (counter, change, entropy))
                if change < self._threshold or counter >= self._maxiter:
                    break
            self.logger.info("")

            self._finalize_propars()
            self.cache.dump("niter", counter, tags="o")
            self.cache.dump("change", change, tags="o")
