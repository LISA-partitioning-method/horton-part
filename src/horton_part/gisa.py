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
"""Gaussian Iterative Stockholder Analysis (GISA) partitioning"""


import logging

import cvxopt
import numpy as np
import quadprog
from scipy.optimize import least_squares

from .core.basis import BasisFuncHelper
from .core.cache import just_once
from .core.iterstock import AbstractISAWPart
from .core.logging import deflist
from .utils import check_pro_atom_parameters

__all__ = [
    "GaussianISAWPart",
    "opt_propars_leastsq_scipy",
    "opt_propars_qp_cvxopt",
    "opt_propars_qp_quadprog",
]

# TODO: the quadratic programming problem can be solved systematically in qpsolvers, where
# different solvers have been implemented.

logger = logging.getLogger(__name__)


class GaussianISAWPart(AbstractISAWPart):
    """Iterative Stockholder Partitioning with Becke-Lebedev grids."""

    name = "gisa"

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
        radius_cutoff=np.inf,
        solver=1,
        solver_kwargs=None,
    ):
        """
        Initial function.

        **Optional arguments:** (that are not defined in :class:`horton_part.core.itertock.AbstractISAWPart`)

        Parameters
        ----------
        solver : int
            The solver id.

        """
        self._solver = solver
        self._solver_kwargs = solver_kwargs or {}
        self._bs_helper = None

        super().__init__(
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            lmax=lmax,
            threshold=threshold,
            maxiter=maxiter,
            inner_threshold=inner_threshold,
            radius_cutoff=radius_cutoff,
        )

    @property
    def bs_helper(self):
        """A basis function helper."""
        if self._bs_helper is None:
            self._bs_helper = BasisFuncHelper.from_function_type()
        return self._bs_helper

    def _init_log_scheme(self):
        deflist(
            logger,
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
            ],
        )
        # biblio.cite(
        #     "verstraelen2012a",
        #     "the use of Gaussian Iterative Stockholder partitioning",
        # )

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
        return self.bs_helper.compute_proatom_dens(self.numbers[iatom], propars, rgrid.points, 1)

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
        output[:] = self.bs_helper.compute_proatom_dens(
            self.numbers[index], populations, self.radial_distances[index], 0
        )

    def _init_propars(self):
        AbstractISAWPart._init_propars(self)
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
            propars[self._ranges[iatom] : self._ranges[iatom + 1]] = self.bs_helper.get_initial(
                self.numbers[iatom]
            )
        self._evaluate_basis_functions()
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
        propars = self.cache.load("propars")[self._ranges[iatom] : self._ranges[iatom + 1]]
        bs_funcs = self.cache.load("bs_funcs", iatom)

        # use truncated grids if local_grid_radius != np.inf
        r_mask = points <= self.radius_cutoff
        points = points[r_mask]
        rho = spherical_average[r_mask]
        weights = rgrid.weights[r_mask]
        bs_funcs = bs_funcs[:, r_mask]
        weights = 4 * np.pi * points**2 * weights

        alphas = self.bs_helper.get_exponent(self.numbers[iatom])
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
        pseudo_population = np.einsum("i,i", weights, rho)
        charges = self.cache.load("charges", alloc=self.natom, tags="o")[0]
        charges[iatom] = self.pseudo_numbers[iatom] - pseudo_population

    def _opt_propars(self, bs_funcs, rho, propars, points, weights, alphas, threshold):
        if callable(self._solver):
            return self._solver(
                bs_funcs,
                rho,
                propars,
                points,
                weights,
                threshold,
                **self._solver_kwargs,
            )

        if self._solver == 1:
            return opt_propars_qp_quadprog(
                bs_funcs, rho, propars, points, weights, alphas, threshold
            )
        elif self._solver == 2:
            return opt_propars_leastsq_scipy(
                bs_funcs, rho, propars, points, weights, alphas, threshold
            )
        elif self._solver == 3:
            return opt_propars_qp_cvxopt(bs_funcs, rho, propars, points, weights, alphas, threshold)
        else:
            raise NotImplementedError

    @just_once
    def _evaluate_basis_functions(self):
        for iatom in range(self.natom):
            rgrid = self.get_rgrid(iatom)
            r = rgrid.points
            nshell = self._ranges[iatom + 1] - self._ranges[iatom]
            bs_funcs = self.cache.load("bs_funcs", iatom, alloc=(nshell, r.size))[0]
            bs_funcs[:, :] = np.array(
                [
                    self.bs_helper.compute_proshell_dens(self.numbers[iatom], ishell, 1.0, r)
                    for ishell in range(nshell)
                ]
            )


def opt_propars_qp_quadprog(bs_funcs, rho, propars, points, weights, alphas, threshold):
    r"""
    Optimize parameters for proatom density functions using quadratic programming.

    .. math::

        N_{Ai} = \int \rho_A(r) \frac{\rho_{Ai}^0(r)}{\rho_A^0(r)} dr


        G = \frac{1}{2} c^T S c - c^T b

        S = 2 \int \zeta(\vec{r}) \zeta(\vec{r}) d\vec{r}
        = \frac{2}{\pi \sqrt{\pi}} \frac{(\alpha_k \alpha_l)^{3/2}}{(\alpha_k + \alpha_l)^{3/2}}

        b = 2 * \int \zeta(\vec{r}) \rho_a(\vec{r}) d\vec{r}

    Parameters
    ----------
    bs_funcs : 2D np.ndarray
        Basis functions array with shape (M, N), where 'M' is the number of basis functions
        and 'N' is the number of grid points.
    rho : 1D np.ndarray
        Spherically-averaged atomic density as a function of radial distance, with shape (N,).
    propars : 1D np.ndarray
        Pro-atom parameters with shape (M). 'M' is the number of basis functions.
    points : 1D np.ndarray
        Radial coordinates of grid points, with shape (N,).
    weights : 1D np.ndarray
        Weights for integration, including the angular part (4πr²), with shape (N,).
    threshold : float
        Convergence threshold for the iterative process.
    alphas : 1D np.ndarray
        The Gaussian exponential coefficients.
    threshold : float
        The convergence threshold of the optimization method.

    Returns
    -------
    1D np.ndarray
        Optimized proatom parameters.

    Raises
    ------
    RuntimeError
        If the inner iteration does not converge.


    """
    nprim = len(propars)
    S = (
        2
        / np.pi**1.5
        * (alphas[:, None] * alphas[None, :]) ** 1.5
        / (alphas[:, None] + alphas[None, :]) ** 1.5
    )

    vec_b = np.zeros(nprim, float)
    for k in range(nprim):
        vec_b[k] = 2 * np.einsum("i,i,i", weights, bs_funcs[k], rho)

    # Construct linear equality or inequality constraints
    matrix_constraint = np.zeros([nprim, nprim + 1])
    # First column : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k)= N_a
    matrix_constraint[:, 0] = np.ones(nprim)
    # Other K_a columns : correspond to the INEQUALITY constraints c_(a,k) >=0
    matrix_constraint[0:nprim, 1 : (nprim + 1)] = np.identity(nprim)
    vector_constraint = np.zeros(nprim + 1)
    # First coefficient : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k) = N_a
    pop = np.einsum("i,i", weights, rho)
    vector_constraint[0] = pop

    propars_qp = quadprog.solve_qp(G=S, a=vec_b, C=matrix_constraint, b=vector_constraint, meq=1)[0]
    check_pro_atom_parameters(propars_qp, total_population=float(pop))
    return propars_qp


def opt_propars_qp_cvxopt(bs_funcs, rho, propars, points, weights, alphas, threshold):
    """
    Optimize pro-atom parameters using quadratic-programming implemented in the `CVXOPT` package.

    Parameters
    ----------
    bs_funcs : 2D np.ndarray
        Basis functions array with shape (M, N), where 'M' is the number of basis functions
        and 'N' is the number of grid points.
    rho : 1D np.ndarray
        Spherically-averaged atomic density as a function of radial distance, with shape (N,).
    propars : 1D np.ndarray
        Pro-atom parameters with shape (M). 'M' is the number of basis functions.
    points : 1D np.ndarray
        Radial coordinates of grid points, with shape (N,).
    weights : 1D np.ndarray
        Weights for integration, including the angular part (4πr²), with shape (N,).
    threshold : float
        Convergence threshold for the iterative process.
    alphas : 1D np.ndarray
        The Gaussian exponential coefficients.
    threshold : float
        The convergence threshold of the optimization method.

    Returns
    -------
    1D np.ndarray
        Optimized proatom parameters.

    Raises
    ------
    RuntimeError
        If the inner iteration does not converge.

    """
    nprim, npt = bs_funcs.shape

    S = (
        2
        / np.pi**1.5
        * (alphas[:, None] * alphas[None, :]) ** 1.5
        / (alphas[:, None] + alphas[None, :]) ** 1.5
    )
    P = cvxopt.matrix(S)

    vec_b = np.zeros((nprim, 1), float)
    for k in range(nprim):
        vec_b[k] = 2 * np.einsum("i,i", weights * bs_funcs[k], rho)
    q = -cvxopt.matrix(vec_b)

    # Linear inequality constraints
    G = cvxopt.matrix(0.0, (nprim, nprim))
    G[:: nprim + 1] = -1.0
    h = cvxopt.matrix(0.0, (nprim, 1))

    # Linear equality constraints
    A = cvxopt.matrix(1.0, (1, nprim))
    pop = np.einsum("i,i", weights, rho)
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
        options={"feastol": threshold},
        # options={"show_progress": log.do_medium, "feastol": threshold},
    )
    new_propars = np.asarray(opt_CVX["x"]).flatten()
    check_pro_atom_parameters(new_propars, total_population=float(pop))
    return new_propars


def opt_propars_leastsq_scipy(bs_funcs, rho, propars, points, weights, alphas, threshold):
    r"""
    Optimize pro-atom parameters using quadratic-programming implemented in the `CVXOPT` package.

    .. math::

        N_{Ai} = \int \rho_A(r) \frac{\rho_{Ai}^0(r)}{\rho_A^0(r)} dr

        G = \frac{1}{2} c^T S c - c^T b

        S = 2 \int \zeta(\vec{r}) \zeta(\vec{r}) d\vec{r}
        = \frac{2}{\pi \sqrt{\pi}} \frac{(\alpha_k \alpha_l)^{3/2}}{(\alpha_k + \alpha_l)^{3/2}}

        b = \int \zeta(\vec{r}) \rho_a(\vec{r}) d\vec{r}

    Parameters
    ----------
    bs_funcs : 2D np.ndarray
        Basis functions array with shape (M, N), where 'M' is the number of basis functions
        and 'N' is the number of grid points.
    rho : 1D np.ndarray
        Spherically-averaged atomic density as a function of radial distance, with shape (N,).
    propars : 1D np.ndarray
        Pro-atom parameters with shape (M). 'M' is the number of basis functions.
    points : 1D np.ndarray
        Radial coordinates of grid points, with shape (N,).
    weights : 1D np.ndarray
        Weights for integration, including the angular part (4πr²), with shape (N,).
    threshold : float
        Convergence threshold for the iterative process.
    alphas : 1D np.ndarray
        The Gaussian exponential coefficients.
    threshold : float
        The convergence threshold of the optimization method.

    Returns
    -------
    1D np.ndarray
        Optimized proatom parameters.

    Raises
    ------
    RuntimeError
        If the inner iteration does not converge.

    """

    def _obj_func(x):
        pro = np.sum(x[:, None] * bs_funcs, axis=0)
        return weights * np.abs(pro - rho)

    res = least_squares(_obj_func, x0=propars, bounds=(0, np.inf), gtol=threshold)
    check_pro_atom_parameters(res.x)
    return res.x
