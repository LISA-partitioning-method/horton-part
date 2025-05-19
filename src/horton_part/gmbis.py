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
"""Generalize Minimal Basis Iterative Stockholder (GMBIS) partitioning"""


import logging

import numpy as np

from .core.logging import deflist
from .mbis import get_nshell, update_propars_atom
from .nlis import NLISWPart, opt_nlis_propars

__all__ = ["GMBISWPart", "get_initial_gmbis_propars"]

logger = logging.getLogger(__name__)


def get_initial_gmbis_propars(number, exp_n_dict, logger=None):
    """Set up initial pro-atom parameters for the NLIS model.

    Parameters
    ----------
    number : int
        Atomic number, denoted by :math:`Z`, of the element for which to generate
        initial pro-atom parameters.
    exp_n_dict: dict
        A dict of exponents for each (number, ishell).
    logger : logging.Logger
        Logger object for capturing logging information.

    Returns
    -------
    np.ndarray
        1D array of initial pro-atom parameters.

    """
    nshell = get_nshell(number)
    propars = np.zeros(3 * nshell, float)
    S0 = 2.0 * number
    if nshell > 1:
        S1 = 2.0
        alpha = (S1 / S0) ** (1.0 / (nshell - 1))
    else:
        alpha = 1.0
    nel_in_shell = np.array([2.0, 8.0, 8.0, 18.0, 18.0, 32.0, 32.0])
    for ishell in range(nshell):
        propars[3 * ishell] = nel_in_shell[ishell]
        propars[3 * ishell + 1] = S0 * alpha**ishell
        if (number, ishell) in exp_n_dict:
            propars[3 * ishell + 2] = exp_n_dict[(number, ishell)]
        else:
            propars[3 * ishell + 2] = 1.0
    propars[-3] = number - propars[:-3:3].sum()
    return propars


class GMBISWPart(NLISWPart):
    """Generalize Minimal Basis Iterative Stockholder (MBIS)"""

    name = "gmbis"

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
        threshold=1e-6,
        maxiter=500,
        inner_threshold=1e-8,
        exp_n_dict=1.0,
        grid_type=1,
        **kwargs,
    ):
        r"""
        Initial function.

        **Optional arguments:** (that are not defined in :class:`horton_part.core.itertock.AbstractISAWPart`)

        Parameters
        ----------
        exp_n_dict : dict
            The power of radial distance :math:`r` in exponential function :math:`exp^{-\alpha r^n}`.

        """
        super().__init__(
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            lmax=lmax,
            logger=logger,
            threshold=threshold,
            maxiter=maxiter,
            inner_threshold=inner_threshold,
            exp_n_dict=exp_n_dict,
            nshell_dict={},
            grid_type=grid_type,
        )

    def _init_log_scheme(self):
        logger.info("Initialized: %s" % self.__class__.__name__)
        deflist(
            logger,
            [
                ("Scheme", "Generalize Minimal Basis Iterative Stockholder (GMBIS)"),
                ("Outer loop convergence threshold", "%.1e" % self._threshold),
                (
                    "Inner loop convergence threshold",
                    "%.1e" % self._inner_threshold,
                ),
                ("Maximum iterations", self._maxiter),
            ],
        )

    def _init_propars(self):
        self._ranges = [0]
        self._nshells = []
        for iatom in range(self.natom):
            nshell = get_nshell(self.numbers[iatom])
            self._ranges.append(self._ranges[-1] + 3 * nshell)
            self._nshells.append(nshell)
        ntotal = self._ranges[-1]
        propars = self.cache.load("propars", alloc=ntotal, tags="o")[0]
        for iatom in range(self.natom):
            propars[self._ranges[iatom] : self._ranges[iatom + 1]] = get_initial_gmbis_propars(
                self.numbers[iatom], self._exp_n_dict, logger=self.logger
            )
        return propars

    def _update_propars_atom(self, iatom):
        update_propars_atom(self, iatom, solver=opt_nlis_propars)
