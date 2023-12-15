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
"""Gaussian Iterative Stockholder Analysis (GISA) partitioning"""


import numpy as np
from .iterstock import ISAWPart
from .log import log, biblio


__all__ = ["IterativeStockholderWPart"]


class IterativeStockholderWPart(ISAWPart):
    """Iterative Stockholder Partitioning with Becke-Lebedev grids"""

    name = "is"
    options = ["lmax", "threshold", "maxiter"]
    linear = False

    def _init_log_scheme(self):
        if log.do_medium:
            log("Initialized: %s" % self.__class__.__name__)
            log.deflist(
                [
                    ("Scheme", "Iterative Stockholder"),
                    ("Outer loop convergence threshold", "%.1e" % self._threshold),
                    (
                        "Inner loop convergence threshold",
                        "%.1e" % self._inner_threshold,
                    ),
                    ("Maximum iterations", self._maxiter),
                    ("lmax", self._lmax),
                ]
            )
        biblio.cite("lillestolen2008", "the use of Iterative Stockholder partitioning")

    def get_rgrid(self, index):
        return self.get_grid(index).rgrid

    def get_proatom_rho(self, index, propars=None):
        """
        Get pro-atom density using the ISA method.

        This function retrieves the density values for a specified atom, based on its index,
        using the ISA method's parameters.

        Parameters
        ----------
        index : int
            The index of an atom in a molecule. Must be a non-negative integer less than the number of atoms.
        propars : np.array, optional
            An array of pro-atom parameters. If None, parameters will be loaded from the cache.

        Returns
        -------
        Tuple: (np.ndarray, None)
            The slice of the pro-atom parameters corresponding to the given atom, and `None` for
            gradient of density.

        """
        if propars is None:
            propars = self.cache.load("propars")
        return propars[self._ranges[index] : self._ranges[index + 1]], None

    def _init_propars(self):
        ISAWPart._init_propars(self)
        self._ranges = [0]
        for index in range(self.natom):
            npoint = self.get_rgrid(index).size
            self._ranges.append(self._ranges[-1] + npoint)
        ntotal = self._ranges[-1]
        return self.cache.load("propars", alloc=ntotal, tags="o")[0]

    def _update_propars_atom(self, index):
        # compute spherical average
        atgrid = self.get_grid(index)
        dens = self.get_moldens(index)
        at_weights = self.cache.load("at_weights", index)
        # avoid too large r
        r = np.clip(atgrid.rgrid.points, 1e-100, 1e10)
        spline = atgrid.spherical_average(at_weights * dens)
        spherical_average = np.clip(spline(r), 1e-100, np.inf)

        # assign as new propars
        propars = self.cache.load("propars")
        propars[self._ranges[index] : self._ranges[index + 1]] = spherical_average

        # compute the new charge
        pseudo_population = atgrid.rgrid.integrate(
            4 * np.pi * r**2 * spherical_average
        )
        charges = self.cache.load("charges", alloc=self.natom, tags="o")[0]
        charges[index] = self.pseudo_numbers[index] - pseudo_population
