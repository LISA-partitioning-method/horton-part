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


import warnings

import numpy as np
import qpsolvers

from horton_part.core import iterstock

from .core.basis import BasisFuncHelper
from .core.cache import just_once
from .core.iterstock import AbstractISAWPart
from .core.logging import deflist
from .utils import check_pro_atom_parameters

__all__ = [
    "GaussianISAWPart",
    "get_proatom_rho",
    "init_propars",
    "evaluate_basis_functions",
]


def get_proatom_rho(part, iatom, propars=None):
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
        propars = part.cache.load("propars")
    rgrid = part.get_rgrid(iatom)
    propars = propars[part._ranges[iatom] : part._ranges[iatom + 1]]
    return part.bs_helper.compute_proatom_dens(part.numbers[iatom], propars, rgrid.points, 1)


def init_propars(part):
    iterstock.init_propars(part)
    part._ranges = [0]
    part._nshells = []
    for iatom in range(part.natom):
        nshell = part.bs_helper.get_nshell(part.numbers[iatom])
        part._ranges.append(part._ranges[-1] + nshell)
        part._nshells.append(nshell)
    ntotal = part._ranges[-1]
    propars = part.cache.load("propars", alloc=ntotal, tags="o")[0]
    propars[:] = 1.0
    # print("Set initial value")
    # print("-" * 80)
    # print(part.pseudo_numbers)
    # print(np.sum(part.pseudo_numbers))
    # print(part.nelec)
    # print("-" * 80)
    for iatom in range(part.natom):
        inits = part.bs_helper.get_initial(part.numbers[iatom])
        # Note: the zero initials are not activated in self-consistent method.
        inits[inits < 1e-4] = 1e-4
        propars[part._ranges[iatom] : part._ranges[iatom + 1]] = (
            inits / np.sum(inits) * part.pseudo_numbers[iatom]
        )
    propars[:] = propars / np.sum(propars) * part.nelec
    part.initial_propars_modified = propars.copy()
    part._evaluate_basis_functions()
    return propars


def evaluate_basis_functions(part):
    for iatom in range(part.natom):
        rgrid = part.get_rgrid(iatom)
        r = rgrid.points
        nshell = part._ranges[iatom + 1] - part._ranges[iatom]
        bs_funcs = part.cache.load("bs_funcs", iatom, alloc=(nshell, r.size))[0]
        bs_funcs[:, :] = np.array(
            [
                part.bs_helper.compute_proshell_dens(part.numbers[iatom], ishell, 1.0, r)
                for ishell in range(nshell)
            ]
        )


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
        logger=None,
        threshold=1e-6,
        maxiter=500,
        inner_threshold=1e-8,
        radius_cutoff=np.inf,
        solver="quadprog",
        solver_options=None,
    ):
        """
        Initial function.

        **Optional arguments:** (that are not defined in :class:`horton_part.core.itertock.AbstractISAWPart`)

        Parameters
        ----------
        solver : str or callable
            The solver name or a callable object.
        solver_options : dict
            The options for different solvers.

        """
        self._solver = solver
        self._solver_options = solver_options or {}
        self._bs_helper = None

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
            self.logger,
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
        if callable(self._solver):
            warnings.warn("Customized solver is used, the argument `inner_threshold` is not used.")
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
        return get_proatom_rho(self, iatom, propars)
        # if propars is None:
        #     propars = self.cache.load("propars")
        # rgrid = self.get_rgrid(iatom)
        # propars = propars[self._ranges[iatom] : self._ranges[iatom + 1]]
        # return self.bs_helper.compute_proatom_dens(self.numbers[iatom], propars, rgrid.points, 1)

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
        return init_propars(self)

    def _update_propars_atom(self, iatom):
        # compute spherical average
        atgrid = self.get_grid(iatom)
        rgrid = atgrid.rgrid
        dens = self.get_moldens(iatom)
        at_weights = self.cache.load("at_weights", iatom)
        spline = atgrid.spherical_average(at_weights * dens)
        points = atgrid.rgrid.points
        spherical_average = spline(points)

        self.cache.dump(f"radial_points_{iatom}", points, tags="o")
        self.cache.dump(f"spherical_average_{iatom}", spherical_average, tags="o")

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
                alphas,
                threshold,
                **self._solver_options,
            )
        else:
            return opt_propars_qp_interface(
                bs_funcs,
                rho,
                propars,
                weights,
                alphas,
                self._solver,
                **self._solver_options,
            )

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


def opt_propars_qp_interface(
    bs_funcs,
    rho,
    propars,
    weights,
    alphas,
    solver="quadprog",
    **solver_options,
):
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
    weights : 1D np.ndarray
        Weights for integration, including the angular part (4πr²), with shape (N,).
    alphas : 1D np.ndarray
        The Gaussian exponential coefficients.
    solver : str
        The name of sovler. See `qpsovler.solve_qp`.

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

    P = (
        2
        / np.pi**1.5
        * (alphas[:, None] * alphas[None, :]) ** 1.5
        / (alphas[:, None] + alphas[None, :]) ** 1.5
    )
    P = (P + P.T) / 2

    q = -2 * np.einsum("i,ni,i->n", weights, bs_funcs, rho)

    # Linear inequality constraints
    G = -np.identity(nprim)
    h = np.zeros((nprim, 1))

    # Linear equality constraints
    A = np.ones((1, nprim))
    pop = np.einsum("i,i", weights, rho)
    b = np.ones((1, 1)) * pop

    # TODO: the initial values are set to zero for all propars and no threshold setting is available.
    result = qpsolvers.solve_qp(
        P,
        q,
        G,
        h,
        A,
        b,
        solver=solver,
        initvals=np.zeros_like(propars),
        **solver_options,
    )
    check_pro_atom_parameters(result, total_population=float(pop))
    return result
