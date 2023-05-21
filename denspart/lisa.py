# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
"""Gaussian Iterative Stockholder Analysis (GISA) partitioning"""


from __future__ import division, print_function
import numpy as np
from horton_grid.log import log
from .gisa import GaussianIterativeStockholderWPart, get_pro_a_k


__all__ = ["LinearIterativeStockholderWPart"]


class LinearIterativeStockholderWPart(GaussianIterativeStockholderWPart):
    name = "lisa"

    def _init_log_scheme(self):
        if log.do_medium:
            log.deflist(
                [
                    ("Scheme", "Linear Iterative Stockholder"),
                    ("Convergence threshold", "%.1e" % self._threshold),
                    ("Maximum iterations", self._maxiter),
                ]
            )
            # biblio.cite(
            #     "Robert2022 the use of Linear Iterative Stockholder ", "partitioning"
            # )

    @staticmethod
    def _opt_propars_by_mbis_opt(rho, propars, rgrid, alphas, threshold):
        r"""
        Optimize parameters for proatom density functions.

        .. math::

            N_{Ai} = \int \rho_A(r) \frac{\rho_{Ai}^0(r)}{\rho_A^0(r)} dr

        Parameters
        ----------
        rho:
            Atomic spherical-average density, i.e.,
            :math:`\langle \rho_A \rangle(|\vec{r}-\vec{r}_A|)`.
        propars:
            Parameters array.
        rgrid:
            Radial grid.
        alphas:
            Exponential coefficients of Gaussian primitive functions.
        threshold:
            Threshold for convergence.

        Returns
        -------

        """
        nprim = len(propars)
        r = rgrid.radii
        oldF = None
        for irep in range(1000):
            # compute the contributions to the pro-atom
            terms = np.array(
                [get_pro_a_k(propars[k], alphas[k], r) for k in range(nprim)]
            )
            pro = terms.sum(axis=0)
            newF = -rgrid.integrate(r**2 * rho * np.log(pro))
            # transform to partitions
            terms *= rho / pro
            # the partitions and the updated parameters
            for k in range(nprim):
                propars[k] = rgrid.integrate(terms[k])
            # check for convergence
            if oldF is None:
                change = 1e100
            else:
                change = np.abs(oldF - newF)
            if change < threshold:
                return propars
            oldF = newF
        log("Not converge, but go ahead!")
        # The initial values could lead to converged issues.
        # assert False
        return propars
