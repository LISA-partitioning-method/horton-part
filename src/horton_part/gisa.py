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
from .basis import BasisFuncHelper, check_pro_atom_parameters
from .cache import just_once


__all__ = ["GaussianIterativeStockholderWPart", "calc_proatom_dens"]


class GaussianIterativeStockholderWPart(ISAWPart):
    """Iterative Stockholder Partitioning with Becke-Lebedev grids"""

    name = "gisa"
    options = ["lmax", "threshold", "maxiter", "solver"]
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
        inner_threshold=1e-8,
        local_grid_radius=8.0,
        solver=1,
        basis_func_type="gauss",
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
        self._solver = solver
        self.func_type = basis_func_type
        self.bs_helper = BasisFuncHelper(basis_func_type)
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
            inner_threshold,
            local_grid_radius,
        )

    def _init_log_scheme(self):
        if log.do_medium:
            log.deflist(
                [
                    ("Scheme", "Gaussian Iterative Stockholder Analysis (GISA)"),
                    ("Outer loop convergence threshold", "%.1e" % self._threshold),
                    (
                        "Inner loop convergence threshold",
                        "%.1e" % self._inner_threshold,
                    ),
                    ("Maximum iterations", self._maxiter),
                    ("lmax", self._lmax),
                    ("Solver", self._solver),
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
        """Get pro-atom density for atom `iatom`.

        If `propars` is `None`, the cache values are used; otherwise, the `propars` are used.

        Parameters
        ----------
        iatom: int
            The index of atom `iatom`.
        propars: np.array
            The pro-atom parameters.

        """
        if propars is None:
            propars = self.cache.load("propars")
        rgrid = self.get_rgrid(iatom)
        propars = propars[self._ranges[iatom] : self._ranges[iatom + 1]]
        alphas = self.bs_helper.load_exponent(self.numbers[iatom])
        return self.bs_helper.compute_proatom_dens(propars, alphas, rgrid.points, 1)

    def eval_proatom(self, index, output, grid):
        """Evaluate function on a local grid.

        The size of the local grid is specified by the radius of the sphere where the local grid is considered.
        For example, when the radius is `np.inf`, the grid corresponds to the whole molecular grid.

        Parameters
        ----------
        index: int
            The index of an atom in the molecule.
        output: np.array
            The size of `output` should be the same as the size of the local grid.
        grid: np.array
            The local grid.

        """
        propars = self.cache.load("propars")
        populations = propars[self._ranges[index] : self._ranges[index + 1]]
        exponents = self.bs_helper.load_exponent(self.numbers[index])
        output[:] = self.bs_helper.compute_proatom_dens(
            populations, exponents, self.radial_dists[index], 0
        )

    def _init_propars(self):
        ISAWPart._init_propars(self)
        self._ranges = [0]
        self._nshells = []
        for iatom in range(self.natom):
            nshell = self.bs_helper.get_nshell(self.numbers[iatom])
            self._ranges.append(self._ranges[-1] + nshell)
            self._nshells.append(nshell)
        ntotal = self._ranges[-1]
        propars = self.cache.load("propars", alloc=ntotal, tags="o")[0]
        propars[:] = 1.0
        for iatom in range(self.natom):
            propars[
                self._ranges[iatom] : self._ranges[iatom + 1]
            ] = self.bs_helper.load_initials(self.numbers[iatom])
        self.compute_integral()
        return propars

    def _update_propars_atom(self, iatom):
        # compute spherical average
        atgrid = self.get_grid(iatom)
        rgrid = atgrid.rgrid
        dens = self.get_moldens(iatom)
        at_weights = self.cache.load("at_weights", iatom)
        spline = atgrid.spherical_average(at_weights * dens)
        points = atgrid.rgrid.points
        spherical_average = spline(points)

        # assign as new propars
        propars = self.cache.load("propars")[
            self._ranges[iatom] : self._ranges[iatom + 1]
        ]
        alphas = self.bs_helper.load_exponent(self.numbers[iatom])
        bs_funcs = self.cache.load("bs_funcs", iatom)

        # use truncated grids if local_grid_radius != np.inf
        r_mask = points <= self.local_grid_radius
        points = points[r_mask]
        rho = spherical_average[r_mask]
        weights = rgrid.weights[r_mask]
        bs_funcs = bs_funcs[:, r_mask]

        # weights of radial grid, without 4 * pi * r**2
        propars[:] = self._opt_propars(
            bs_funcs,
            rho,
            propars.copy(),
            points,
            weights,
            alphas,
            self._inner_threshold,
        )

        # compute the new charge
        pseudo_population = np.einsum("i,i,i", weights, 4 * np.pi * points**2, rho)
        charges = self.cache.load("charges", alloc=self.natom, tags="o")[0]
        charges[iatom] = self.pseudo_numbers[iatom] - pseudo_population

    def _opt_propars(self, bs_funcs, rho, propars, points, weights, alphas, threshold):
        if self._solver == 1:
            return _constrained_least_squares_quadprog(
                bs_funcs, rho, propars, points, weights, alphas, threshold
            )
        elif self._solver == 2:
            return _constrained_least_squares(
                bs_funcs, rho, propars, points, weights, alphas, threshold
            )
        elif self._solver == 3:
            return _constrained_least_cvxopt(
                bs_funcs, rho, propars, points, weights, alphas, threshold
            )
        elif self._solver == 0:
            return _solver_comparison(
                bs_funcs, rho, propars, points, weights, alphas, threshold
            )
        else:
            raise NotImplementedError

    @just_once
    def compute_integral(self):
        for a in range(self.natom):
            rgrid = self.get_rgrid(a)
            r = rgrid.points
            # r = r[r <= self.local_grid_radius]

            nprim = self._ranges[a + 1] - self._ranges[a]
            alphas = self.bs_helper.load_exponent(self.numbers[a])

            bs_funcs = self.cache.load("bs_funcs", a, alloc=(nprim, r.size))[0]
            bs_funcs[:, :] = np.array(
                [
                    self.bs_helper.compute_proshell_dens(1.0, alphas[k], r)
                    for k in range(nprim)
                ]
            )


def calc_proatom_dens(propars, bs_funcs):
    shells = propars[:, None] * bs_funcs
    return shells.sum(axis=0)


def _constrained_least_squares_quadprog(
    bs_funcs, rho, propars, points, weights, alphas, threshold
):
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
    r = points
    S = (
        2
        / np.pi**1.5
        * (alphas[:, None] * alphas[None, :]) ** 1.5
        / (alphas[:, None] + alphas[None, :]) ** 1.5
    )

    vec_b = np.zeros(nprim, float)
    for k in range(nprim):
        vec_b[k] = 2 * np.einsum(
            "i,i,i->", 4 * np.pi * r**2 * weights, bs_funcs[k], rho
        )

    # Construct linear equality or inequality constraints
    matrix_constraint = np.zeros([nprim, nprim + 1])
    # First column : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k)= N_a
    matrix_constraint[:, 0] = np.ones(nprim)
    # Other K_a columns : correspond to the INEQUALITY constraints c_(a,k) >=0
    matrix_constraint[0:nprim, 1 : (nprim + 1)] = np.identity(nprim)
    vector_constraint = np.zeros(nprim + 1)
    # First coefficient : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k) = N_a
    pop = np.einsum("i,i->", 4 * np.pi * r**2 * weights, rho)
    vector_constraint[0] = pop

    propars_qp = quadprog.solve_qp(
        G=S, a=vec_b, C=matrix_constraint, b=vector_constraint, meq=1
    )[0]
    check_pro_atom_parameters(propars_qp, total_population=float(pop))
    return propars_qp


def _constrained_least_squares(
    bs_funcs, rho, propars, points, weights, alphas, threshold
):
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

    def f(propars):
        # rho_test = bs_helper.compute_proatom_dens(args, alphas, r, 0)
        pro = calc_proatom_dens(propars, bs_funcs)
        # This is integrand function corresponding to the difference of number of electrons.
        return 4 * np.pi * weights * points**2 * np.abs(pro - rho)

    res = least_squares(
        f,
        x0=propars,
        bounds=(0, np.inf),
        verbose=2 if log.level >= 2 else log.level,
        gtol=threshold,
    )
    check_pro_atom_parameters(res.x)
    return res.x


def _constrained_least_cvxopt(
    bs_funcs, rho, propars, points, weights, alphas, threshold
):
    nprim, npt = bs_funcs.shape
    r = points

    S = (
        2
        / np.pi**1.5
        * (alphas[:, None] * alphas[None, :]) ** 1.5
        / (alphas[:, None] + alphas[None, :]) ** 1.5
    )
    P = cvxopt.matrix(S)

    vec_b = np.zeros((nprim, 1), float)
    for k in range(nprim):
        vec_b[k] = 2 * np.einsum(
            "i,i->", 4 * np.pi * r**2 * weights * bs_funcs[k], rho
        )
    q = -cvxopt.matrix(vec_b)

    # Linear inequality constraints
    G = cvxopt.matrix(0.0, (nprim, nprim))
    G[:: nprim + 1] = -1.0
    h = cvxopt.matrix(0.0, (nprim, 1))

    # Linear equality constraints
    A = cvxopt.matrix(1.0, (1, nprim))
    pop = np.einsum("i,i->", 4 * np.pi * r**2 * weights, rho)
    b = cvxopt.matrix(pop, (1, 1))

    # initial_values = cvxopt.matrix(np.array([1.0] * nprim).reshape((nprim, 1)))
    opt_CVX = cvxopt.solvers.qp(
        P,
        q,
        G,
        h,
        A,
        b,
        initvals=propars,
        options={"show_progress": log.do_medium, "feastol": threshold},
    )
    new_propars = np.asarray(opt_CVX["x"]).flatten()
    check_pro_atom_parameters(new_propars, total_population=float(pop))
    return new_propars


def _solver_comparison(bs_funcs, rho, propars, points, weights, alphas, threshold):
    propars_qp = _constrained_least_squares_quadprog(
        bs_funcs, rho, propars, points, weights, alphas, threshold
    )
    print("propars_qp:")
    propars_qp = np.clip(propars_qp, 0, np.inf)
    print(propars_qp)

    propars_lsq = _constrained_least_squares(
        bs_funcs, rho, propars, points, weights, alphas, threshold, x0=propars_qp
    )
    print("propars_lsq:")
    print(propars_lsq)

    propars_cvxopt = _constrained_least_cvxopt(
        bs_funcs, rho, propars, points, weights, alphas, threshold
    )
    print("propars_cvxopt:")
    print(propars_cvxopt)
    return propars_cvxopt
