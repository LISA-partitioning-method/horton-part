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


import numpy as np
from .wrapper import log, biblio
import quadprog
from .stockholder import StockholderWPart
from .iterstock import IterativeProatomMixin


__all__ = ["GaussianIterativeStockholderWPart", "get_pro_a_k"]


def get_alpha(number):
    """The exponents used for primitive Gaussian functions of each element."""
    param_dict = {
        1: np.array([5.672, 1.505, 0.5308, 0.2204]),
        6: np.array([148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]),
        7: np.array([178.0, 52.42, 19.87, 1.276, 0.6291, 0.2857]),
        8: np.array([220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]),
    }
    if number in param_dict:
        return param_dict[number]
    else:
        raise NotImplementedError


def get_nprim(number):
    """The number of primitive Gaussian functions used for each element."""
    return len(get_alpha(number))


def get_pro_a_k(D_k, alpha_k, r, nderiv=0):
    """The primitive Gaussian function with exponent of `alpha_k`."""
    f = D_k * (alpha_k / np.pi) ** 1.5 * np.exp(-alpha_k * r**2)
    if nderiv == 0:
        return f
    elif nderiv == 1:
        return -2 * alpha_k * r * f
    else:
        raise NotImplementedError


class GaussianIterativeStockholderWPart(IterativeProatomMixin, StockholderWPart):
    """Iterative Stockholder Partitioning with Becke-Lebedev grids"""

    name = "gisa"
    options = ["lmax", "threshold", "maxiter", "obj_fn_type"]
    linear = False

    def __init__(
        self,
        coordinates,
        numbers,
        pseudo_numbers,
        grid,
        moldens,
        spindens=None,
        lmax=3,
        threshold=1e-6,
        maxiter=500,
        obj_fn_type=1,
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
        """
        self._threshold = threshold
        self._maxiter = maxiter
        self._obj_fn_type = obj_fn_type
        StockholderWPart.__init__(
            self,
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            True,
            lmax,
        )

    def _init_log_scheme(self):
        if log.do_medium:
            log.deflist(
                [
                    ("Scheme", "Iterative Stockholder"),
                    ("Convergence threshold", "%.1e" % self._threshold),
                    ("Maximum iterations", self._maxiter),
                ]
            )
            biblio.cite(
                "lillestolen2008", "the use of Iterative Stockholder partitioning"
            )

    def get_rgrid(self, index):
        """Load radial grid."""
        return self.get_grid(index).rgrid

    def get_proatom_rho(self, iatom, propars=None):
        """Get proatom density for atom `iatom`."""
        if propars is None:
            propars = self.cache.load("propars")
        rgrid = self.get_rgrid(iatom)
        r = rgrid.radii
        y = np.zeros(len(r), float)
        d = np.zeros(len(r), float)
        propars = propars[self._ranges[iatom] : self._ranges[iatom + 1]]
        alphas = get_alpha(self.numbers[iatom])
        for k in range(self._nprims[iatom]):
            D_k, alpha_k = propars[k], alphas[k]
            y += get_pro_a_k(D_k, alpha_k, r, 0)
            d += get_pro_a_k(D_k, alpha_k, r, 1)
        return y, d

    def _init_propars(self):
        IterativeProatomMixin._init_propars(self)
        self._ranges = [0]
        self._nprims = []
        for iatom in range(self.natom):
            nprim = get_nprim(self.numbers[iatom])
            self._ranges.append(self._ranges[-1] + nprim)
            self._nprims.append(nprim)
        ntotal = self._ranges[-1]
        propars = self.cache.load("propars", alloc=ntotal, tags="o")[0]
        propars[:] = 1.0
        for iatom in range(self.natom):
            propars[
                self._ranges[iatom] : self._ranges[iatom + 1]
            ] = self._get_initial_propars(self.numbers[iatom])
        return propars

    def _update_propars_atom(self, iatom):
        # compute spherical average
        atgrid = self.get_grid(iatom)
        rgrid = atgrid.rgrid
        dens = self.get_moldens(iatom)
        at_weights = self.cache.load("at_weights", iatom)
        spherical_average = np.clip(
            atgrid.get_spherical_average(at_weights, dens), 1e-100, np.inf
        )

        # assign as new propars
        propars = self.cache.load("propars")[
            self._ranges[iatom] : self._ranges[iatom + 1]
        ]
        alphas = get_alpha(self.numbers[iatom])
        propars[:] = self._opt_propars(
            spherical_average, propars.copy(), rgrid, alphas, self._threshold
        )

        # compute the new charge
        pseudo_population = atgrid.rgrid.integrate(spherical_average)
        charges = self.cache.load("charges", alloc=self.natom, tags="o")[0]
        charges[iatom] = self.pseudo_numbers[iatom] - pseudo_population

    @staticmethod
    def _get_initial_propars(number):
        """Create initial parameters for proatom density functions."""
        nprim = get_nprim(number)
        # return np.ones(nprim, float) / nprim
        return np.ones(nprim, float) / nprim

    def _opt_propars(self, rho, propars, rgrid, alphas, threshold):
        return self._constrained_least_squares(rho, propars, rgrid, alphas, threshold)

    @staticmethod
    def _constrained_least_squares(rho, propars, rgrid, alphas, threshold):
        r"""
        Optimize parameters for proatom density functions.

        .. math::

            N_{Ai} = \int \rho_A(r) \frac{\rho_{Ai}^0(r)}{\rho_A^0(r)} dr


            G = \frac{1}{2} c^T S c - c^T b

            S = 2 \int \zeta(\vec{r}) \zeta(\vec{r}) d\vec{r}
            = \frac{2}{\pi \sqrt{\pi}} \frac{(\alpha_k \alpha_l)^{3/2}}{(\alpha_k + \alpha_l)^{3/2}}

            b = \int \zeta(\vec{r}) \rho_a(\vec{r}) d\vec{r}

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
        gauss_funcs = np.array([get_pro_a_k(1.0, alphas[k], r) for k in range(nprim)])

        # compute the contributions to the pro-atom
        # terms = np.array([get_pro_a_k(propars[k], alphas[k], r) for k in range(nprim)])
        # pro = terms.sum(axis=0)

        S = (
            1
            / np.pi**1.5
            * (alphas[:, None] * alphas[None, :]) ** 1.5
            / (alphas[:, None] + alphas[None, :]) ** 1.5
        )

        vec_b = np.zeros(nprim, float)
        for k in range(nprim):
            vec_b[k] = rgrid.integrate(gauss_funcs[k], rho)

        # METHOD 2 : using Python quadratic programming optimization routines
        # (linear equality or inequality constraints)
        matrix_constraint = np.zeros([nprim, nprim + 1])
        # First column : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k)= N_a
        matrix_constraint[:, 0] = np.ones(nprim)
        # Other K_a columns : correspond to the INEQUALITY constraints c_(a,k) >=0
        matrix_constraint[0:nprim, 1 : (nprim + 1)] = np.identity(nprim)

        vector_constraint = np.zeros(nprim + 1)

        # First coefficient : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k) = N_a
        N_a = rgrid.integrate(rho)
        vector_constraint[0] = N_a

        propars = quadprog.solve_qp(
            G=S, a=vec_b, C=matrix_constraint, b=vector_constraint, meq=1
        )[0]

        print(propars)
        print("charges:", np.sum(propars) - N_a)

        return propars
