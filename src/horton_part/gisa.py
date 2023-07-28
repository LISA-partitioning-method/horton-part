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
from scipy.optimize import least_squares
import quadprog
import cvxopt

# from cvxopt.solvers import qp
from .iterstock import ISAWPart
from .log import log, biblio


__all__ = ["GaussianIterativeStockholderWPart", "get_pro_a_k"]


def get_alpha(number, nb_exp=6):
    """The exponents used for primitive Gaussian functions of each element."""
    assert isinstance(number, (int, np.int_)) and isinstance(nb_exp, (int, np.int_))
    param_dict = {
        1: np.array([5.672, 1.505, 0.5308, 0.2204]),
        3: np.array([0.008, 0.042, 0.1214, 2.8398, 10.7499, 87.8413]),
        6: np.array([148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]),
        7: np.array([178.0, 52.42, 19.87, 1.276, 0.6291, 0.2857]),
        8: np.array([220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]),
        # Even we use O atom data as Cl atom data, we can still obtain a reasonable result for LISA model for
        # ClO- molecule. This means it needs small exponential coefficients.
        # use O atom data for Cl
        # 17: np.array([220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]),
        # use C atom data for Cl
        # 17: np.array([148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]),
        17: np.array(
            [
                0.0955,
                0.2188,
                0.5903,
                0.7801,
                8.8711,
                19.2626,
                164.0007,
                373.7075,
                591.4187,
            ]
        ),
    }
    if number in param_dict:
        return param_dict[number]
    else:
        # use MBIS initial exponential coefficients as suggested by GISA.
        # The difference between this implentation and GISA is that Bohr radius is removed.
        # This is because atomic unit is alwasy used in cucrrent implementation.
        # The Bohr radius included in original implementation of GISA in
        # https://github.com/rbenda/ISA_multipoles is a typo.
        # (also see J. Chem. Phys. 156, 164107 (2022))
        #
        # For Li and  Cl, nb_exp = 6
        # a0 = 0.529177
        a0 = 1  # we use atomic unit
        # TODO: in Toon's paper, these coefficients are optimized by fitting Hirshfeld-I pro-atomic density
        return np.array(
            [
                2 * number ** (1 - ((i - 1) / (nb_exp - 1))) / a0
                for i in range(1, 1 + nb_exp)
            ]
        )


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


def get_proatom_rho(r, prefactors, alphas):
    """Get proatom density for atom `iatom`."""
    nprim = len(prefactors)
    # r = rgrid.radii
    y = np.zeros(len(r), float)
    d = np.zeros(len(r), float)
    for k in range(nprim):
        D_k, alpha_k = prefactors[k], alphas[k]
        y += get_pro_a_k(D_k, alpha_k, r, 0)
        d += get_pro_a_k(D_k, alpha_k, r, 1)
    return y, d


class GaussianIterativeStockholderWPart(ISAWPart):
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
        self._obj_fn_type = obj_fn_type
        ISAWPart.__init__(
            self,
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            True,
            lmax,
            threshold,
            maxiter,
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
                "verstraelen2012a",
                "the use of Gaussian Iterative Stockholder partitioning",
            )

    def get_rgrid(self, index):
        """Load radial grid."""
        return self.get_grid(index).rgrid

    def get_proatom_rho(self, iatom, propars=None):
        """Get proatom density for atom `iatom`."""
        if propars is None:
            propars = self.cache.load("propars")
        rgrid = self.get_rgrid(iatom)
        r = rgrid.points
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
        ISAWPart._init_propars(self)
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
        spline = atgrid.spherical_average(at_weights * dens)
        # avoid too large r
        r = np.clip(atgrid.rgrid.points, 1e-100, 1e10)
        spherical_average = np.clip(spline(r), 1e-100, np.inf)

        # assign as new propars
        propars = self.cache.load("propars")[
            self._ranges[iatom] : self._ranges[iatom + 1]
        ]
        alphas = get_alpha(self.numbers[iatom])
        propars[:] = self._opt_propars(
            spherical_average, propars.copy(), rgrid, alphas, self._threshold
        )

        # compute the new charge
        pseudo_population = atgrid.rgrid.integrate(
            4 * np.pi * r**2 * spherical_average
        )
        charges = self.cache.load("charges", alloc=self.natom, tags="o")[0]
        charges[iatom] = self.pseudo_numbers[iatom] - pseudo_population

    @staticmethod
    def _get_initial_propars(number):
        """Create initial parameters for proatom density functions."""
        nprim = get_nprim(number)
        # from https://github.com/rbenda/ISA_multipoles.
        if number == 17:
            return np.ones(nprim, float) * 17.5 / nprim
        elif number == 8:
            return np.ones(nprim, float) * 8.5 / nprim
        else:
            return np.ones(nprim, float) * (number + 0.5) / nprim

    def _opt_propars(self, rho, propars, rgrid, alphas, threshold):
        if self._obj_fn_type == 1:
            return self._constrained_least_squares_quadprog(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._obj_fn_type == 2:
            return self._constrained_least_squares(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._obj_fn_type == 3:
            return self._constrained_least_cvxopt(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._obj_fn_type == 0:
            return self._solver_comparison(rho, propars, rgrid, alphas, threshold)
        else:
            raise NotImplementedError

    @staticmethod
    def _constrained_least_squares_quadprog(rho, propars, rgrid, alphas, threshold):
        r"""
        Optimize parameters for proatom density functions.

        .. math::

            N_{Ai} = \int \rho_A(r) \frac{\rho_{Ai}^0(r)}{\rho_A^0(r)} dr


            G = \frac{1}{2} c^T S c - c^T b

            S = 2 \int \zeta(\vec{r}) \zeta(\vec{r}) d\vec{r}
            = \frac{2}{\pi \sqrt{\pi}} \frac{(\alpha_k \alpha_l)^{3/2}}{(\alpha_k + \alpha_l)^{3/2}}

            b = 2 * \int \zeta(\vec{r}) \rho_a(\vec{r}) d\vec{r}

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
        r = rgrid.points
        # avoid too large r
        r = np.clip(r, 1e-100, 1e10)
        gauss_funcs = np.array([get_pro_a_k(1.0, alphas[k], r) for k in range(nprim)])
        S = (
            2
            / np.pi**1.5
            * (alphas[:, None] * alphas[None, :]) ** 1.5
            / (alphas[:, None] + alphas[None, :]) ** 1.5
        )

        vec_b = np.zeros(nprim, float)
        for k in range(nprim):
            vec_b[k] = 2 * rgrid.integrate(4 * np.pi * r**2, gauss_funcs[k], rho)

        # Construct linear equality or inequality constraints
        matrix_constraint = np.zeros([nprim, nprim + 1])
        # First column : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k)= N_a
        matrix_constraint[:, 0] = np.ones(nprim)
        # Other K_a columns : correspond to the INEQUALITY constraints c_(a,k) >=0
        matrix_constraint[0:nprim, 1 : (nprim + 1)] = np.identity(nprim)
        vector_constraint = np.zeros(nprim + 1)
        # First coefficient : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k) = N_a
        vector_constraint[0] = rgrid.integrate(4 * np.pi * r**2, rho)

        propars_qp = quadprog.solve_qp(
            G=S, a=vec_b, C=matrix_constraint, b=vector_constraint, meq=1
        )[0]
        return propars_qp

    @staticmethod
    def _constrained_least_squares(rho, propars, rgrid, alphas, threshold, x0=None):
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
        r = rgrid.points
        # avoid too large r
        r = np.clip(r, 1e-100, 1e10)
        weights = rgrid.weights
        nelec = rgrid.integrate(4 * np.pi * r**2, rho)

        def f(args):
            rho_test, _ = get_proatom_rho(r, args, alphas)
            # This is integrand function corresponding to the difference of number of electrons.
            obj_func = 4 * np.pi * np.abs(rho_test - rho) * weights * r**2
            return obj_func

        if x0 is None:
            x0 = [nelec / nprim] * nprim
        res = least_squares(
            f,
            x0=x0,
            bounds=(0, np.inf),
            verbose=2,
        )
        return res.x

    @staticmethod
    def _constrained_least_cvxopt(rho, propars, rgrid, alphas, threshold):
        nprim = len(propars)
        r = rgrid.points
        # avoid too large r
        r = np.clip(r, 1e-100, 1e10)
        gauss_funcs = np.array([get_pro_a_k(1.0, alphas[k], r) for k in range(nprim)])
        S = (
            2
            / np.pi**1.5
            * (alphas[:, None] * alphas[None, :]) ** 1.5
            / (alphas[:, None] + alphas[None, :]) ** 1.5
        )
        P = cvxopt.matrix(S)

        vec_b = np.zeros((nprim, 1), float)
        for k in range(nprim):
            vec_b[k] = 2 * rgrid.integrate(4 * np.pi * r**2, gauss_funcs[k], rho)
        q = -cvxopt.matrix(vec_b)

        # Linear inequality constraints
        G = cvxopt.matrix(0.0, (nprim, nprim))
        G[:: nprim + 1] = -1.0
        # G = -cvxopt.matrix(np.identity(nprim))
        h = cvxopt.matrix(0.0, (nprim, 1))

        # Linear equality constraints
        A = cvxopt.matrix(1.0, (1, nprim))
        Na = rgrid.integrate(4 * np.pi * r**2, rho)
        b = cvxopt.matrix(Na, (1, 1))

        # initial_values = cvxopt.matrix(np.array([1.0] * nprim).reshape((nprim, 1)))
        opt_CVX = cvxopt.solvers.qp(P, q, G, h, A, b)
        new_propars = np.asarray(opt_CVX["x"]).flatten()
        return new_propars

    @staticmethod
    def _solver_comparison(rho, propars, rgrid, alphas, threshold):
        propars_qp = (
            GaussianIterativeStockholderWPart._constrained_least_squares_quadprog(
                rho, propars, rgrid, alphas, threshold
            )
        )
        print("propars_qp:")
        propars_qp = np.clip(propars_qp, 0, np.inf)
        print(propars_qp)

        propars_lsq = GaussianIterativeStockholderWPart._constrained_least_squares(
            rho, propars, rgrid, alphas, threshold, x0=propars_qp
        )
        print("propars_lsq:")
        print(propars_lsq)

        propars_cvxopt = GaussianIterativeStockholderWPart._constrained_least_cvxopt(
            rho, propars, rgrid, alphas, threshold
        )
        print("propars_cvxopt:")
        print(propars_cvxopt)
        return propars_cvxopt
