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
"""Iterative Hirshfeld (HI) partitioning"""


import numpy as np

from .core.cache import just_once
from .core.iterstock import AbstractISAWPart
from .core.logging import deflist
from .hirshfeld import check_proatomdb, do_dispersion

# from .core.log import log, biblio


__all__ = ["HirshfeldIWPart"]


class HirshfeldIWPart(AbstractISAWPart):
    """Iterative Hirshfeld partitioning with Becke-Lebedev grids"""

    name = "hi"

    def __init__(
        self,
        coordinates,
        numbers,
        pseudo_numbers,
        grid,
        moldens,
        proatomdb,
        spindens=None,
        lmax=3,
        logger=None,
        threshold=1e-6,
        maxiter=500,
        grid_type=1,
        **kwargs,
    ):
        """
        **Arguments:** (that are not defined in ``WPart``)

        proatomdb
             In instance of ProAtomDB that contains all the reference atomic
             densities.

        **Optional arguments:** (that are not defined in ``WPart``)

        threshold
             The procedure is considered to be converged when the maximum
             change of the charges between two iterations drops below this
             threshold.

        maxiter
             The maximum number of iterations. If no convergence is reached
             in the end, no warning is given.
        """
        check_proatomdb(numbers, pseudo_numbers, proatomdb)
        self._proatomdb = proatomdb

        AbstractISAWPart.__init__(
            self,
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            # proatomdb,
            spindens,
            lmax,
            logger,
            threshold,
            maxiter,
            grid_type=grid_type,
        )

    def _init_log_scheme(self):
        self.logger.info("Initialized: %s" % self.__class__.__name__)
        deflist(
            self.logger,
            [
                ("Scheme", "Hirshfeld-I"),
                ("Convergence threshold", "%.1e" % self._threshold),
                ("Maximum iterations", self._maxiter),
                ("Proatomic DB", self._proatomdb),
            ],
        )
        # biblio.cite("bultinck2007", "the use of Hirshfeld-I partitioning")

    @property
    def proatomdb(self):
        """Database of atomic pro-atom density."""
        return self._proatomdb

    def get_rgrid(self, index):
        number = self.numbers[index]
        return self.proatomdb.get_rgrid(number)

    def get_interpolation_info(self, i, charges=None):
        if charges is None:
            charges = self.cache.load("charges")
        target_charge = charges[i]
        icharge = int(np.floor(target_charge))
        x = target_charge - icharge
        return icharge, x

    def get_proatom_rho(self, iatom, charges=None, **kwargs):
        icharge, x = self.get_interpolation_info(iatom, charges)
        # check if icharge record should exist
        pseudo_pop = self.pseudo_numbers[iatom] - icharge
        number = self.numbers[iatom]
        if pseudo_pop == 1 or x == 0.0:
            return self.proatomdb.get_rho(number, {icharge: 1 - x}, do_deriv=True)
        elif pseudo_pop > 1:
            return self.proatomdb.get_rho(number, {icharge: 1 - x, icharge + 1: x}, do_deriv=True)
        elif pseudo_pop <= 0:
            raise ValueError("Requesting a pro-atom with a negative (pseudo) population")

    def get_somefn(self, index, spline, key, label, grid):
        key = key + (index, id(grid))
        result, new = self.cache.load(*key, alloc=grid.size)
        if new:
            self.eval_spline(index, spline, result, grid, label)
        return result

    def get_isolated(self, index, charge, grid):
        number = self.numbers[index]
        spline = self.proatomdb.get_spline(number, charge)
        return self.get_somefn(index, spline, ("isolated", charge), "isolated q=%+i" % charge, grid)

    def eval_proatom(self, index, output, grid):
        # Greedy version of eval_proatom
        icharge, x = self.get_interpolation_info(index)
        output[:] = self.get_isolated(index, icharge, grid)
        output *= 1 - x
        pseudo_pop = self.pseudo_numbers[index] - icharge
        if pseudo_pop > 1 and x != 0.0:
            output += self.get_isolated(index, icharge + 1, grid) * x
        elif pseudo_pop <= 0:
            raise ValueError("Requesting a pro-atom with a negative (pseudo) population")
        output += 1e-100

    def _init_propars(self):
        # AbstractISAWPart._init_propars(self)
        charges = self.cache.load("charges", alloc=self.natom, tags="o")[0]
        self.cache.dump("propars", charges, tags="o")
        return charges

    def _update_propars_atom(self, index):
        # Compute population
        pseudo_population = self.compute_pseudo_population(index)

        # Store charge
        charges = self.cache.load("charges")
        charges[index] = self.pseudo_numbers[index] - pseudo_population

    @just_once
    def do_dispersion(self):
        do_dispersion(self)
