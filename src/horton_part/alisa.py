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
"""
Module for Linear Iterative Stockholder Analysis (L-ISA) partitioning scheme.
"""

import logging
import warnings

import cvxopt
import numpy as np
from scipy.linalg import LinAlgWarning, solve
from scipy.optimize import SR1, LinearConstraint, minimize
from scipy.sparse import SparseEfficiencyWarning

from .algo.cdiis import cdiis
from .algo.diis import diis
from .algo.quasi_newton import bfgs
from .core.basis import (
    AnalyticBasisFuncHelper,
    ExpBasisFuncHelper,
    NumericBasisFuncHelper,
)
from .core.logging import deflist
from .gisa import GaussianISAWPart
from .utils import (
    check_pro_atom_parameters_neg_pars,
    check_pro_atom_parameters_non_neg_pars,
    compute_quantities,
)

# Suppress specific warning
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


__all__ = [
    "solver_sc",
    "solver_diis",
    "solver_newton",
    "solver_m_newton",
    "solver_trust_region",
    "solver_cvxopt",
    "solver_cdiis",
    "setup_bs_helper",
    "LinearISAWPart",
]


def solver_cvxopt(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    logger,
    density_cutoff,
    negative_cutoff,
    population_cutoff,
    # Options for solver.
    allow_neg_params=False,
    **cvxopt_options,
):
    """
    Optimize parameters for pro-atom density functions using convex optimization method.

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
    logger : logging.Logger or None
        The log object.
    density_cutoff : float
        Density values below this cutoff are considered invalid.
    negative_cutoff : float
        The value less than `negative_cutoff` is treated as negative.
    population_cutoff : float
        The criteria for the difference between the sum of propars and reference population.
    allow_neg_params : bool (optional)
        Whether negative parameters are allowed. The default is `False`.
    cvxopt_options: dict
        Other options for the cvxopt solver.

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
    if allow_neg_params:
        matrix_constraint_ineq = None
        vector_constraint_ineq = None
    else:
        matrix_constraint_ineq = -cvxopt.matrix(np.identity(nprim))
        vector_constraint_ineq = cvxopt.matrix(0.0, (nprim, 1))

    pop = np.einsum("i,i", weights, rho)
    matrix_constraint_eq = cvxopt.matrix(1.0, (1, nprim))
    vector_constraint_eq = cvxopt.matrix(pop, (1, 1))

    def obj_func(x=None, z=None):
        """The local objective function for an atom."""
        if x is None:
            return 0, cvxopt.matrix(propars[:])

        pro_shells, pro, sick, ratio, ln_ratio = compute_quantities(
            rho, x, bs_funcs, density_cutoff
        )
        f = np.einsum("i,i,i", weights, rho, ln_ratio)

        # compute gradient
        grad = weights * ratio
        df = -np.einsum("j,ij->i", grad, bs_funcs)
        df = cvxopt.matrix(df.reshape((1, nprim)))
        if z is None:
            return f, df

        # compute hessian
        hess = np.divide(grad, pro, out=np.zeros_like(grad), where=~sick)
        d2f = np.einsum("k,ik,jk->ij", hess, bs_funcs, bs_funcs)
        d2f = z[0] * cvxopt.matrix(d2f)
        return f, df, d2f

    cvxopt_options = cvxopt_options or {}
    if len(cvxopt_options) == 0:
        show_progress = 0
        if logger.level <= logging.DEBUG:
            show_progress = 3
        options = {"show_progress": show_progress, "feastol": threshold}
    else:
        options = cvxopt_options

    opt_CVX = cvxopt.solvers.cp(
        obj_func,
        G=matrix_constraint_ineq,
        h=vector_constraint_ineq,
        A=matrix_constraint_eq,
        b=vector_constraint_eq,
        options=options,
    )

    if opt_CVX["status"] == "optimal":
        optimized_res = opt_CVX["x"]
        new_propars = np.asarray(optimized_res).flatten()
        check_pro_atom_parameters_non_neg_pars(
            new_propars,
            basis_functions=np.asarray(bs_funcs),
            logger=logger,
            total_population=float(pop),
            negative_cutoff=negative_cutoff,
            population_cutoff=population_cutoff,
        )
        return new_propars
    else:
        logger.error("CVXOPT not converged!")


def solver_sc(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    logger,
    density_cutoff,
    negative_cutoff,
    population_cutoff,
    # Options for solver.
    max_niter_inner=100000,
):
    r"""
    Optimize parameters for proatom density functions using a self-consistent (SC) method.

    This approach analytically computes the parameters, aiming to yield results comparable to
    those obtained via L-ISA algorithms, which require non-negative parameters.

     .. math::

        N_{Ai} = \int \rho_A(r) \frac{\rho_{Ai}^0(r)}{\rho_A^0(r)} dr

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
    logger : logging.Logger or None
        The log object.
    density_cutoff : float
        Density values below this cutoff are considered invalid.
    negative_cutoff : float
        The value less than `negative_cutoff` is treated as negative.
    population_cutoff : float
        The criteria for the difference between the sum of propars and reference population.
    max_niter_inner : int
        The maximum number of inner iterations.

    Returns
    -------
    1D np.ndarray
        Optimized proatom parameters.

    Raises
    ------
    RuntimeError
        If the inner iteration does not converge.

    Notes
    -----
    The method iteratively optimizes the proatom density function parameters.
    In each iteration, the basis functions and current parameters are used to compute
    updated parameters, assessing convergence against the specified threshold.

    """
    logger.debug("            Iter.    Change    ")
    logger.debug("            -----    ------    ")
    oldpro = None
    for irep in range(int(float(max_niter_inner))):
        pro_shells, pro, sick, ratio, _ = compute_quantities(
            rho, propars, bs_funcs, density_cutoff, do_ln_ratio=False
        )
        # the partitions and the updated parameters
        propars[:] = np.einsum("p,ip->i", weights, pro_shells * ratio)

        # check for convergence
        if oldpro is None:
            change = 1e100
        else:
            error = oldpro - pro
            change = np.sqrt(np.einsum("i,i,i", weights, error, error))

        logger.debug(f"            {irep+1:<4}    {change:.3e}")
        if change < threshold:
            pop = np.einsum("i,i", weights, rho)
            check_pro_atom_parameters_non_neg_pars(
                propars,
                basis_functions=np.asarray(bs_funcs),
                logger=logger,
                total_population=float(pop),
                negative_cutoff=negative_cutoff,
                population_cutoff=population_cutoff,
            )
            return propars
        oldpro = pro
    logger.warning("Warning: Inner iteration is not converge!")
    return propars


def solver_sc_1_iter(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    logger,
    density_cutoff,
    negative_cutoff,
    population_cutoff,
    # Options for solver.
):
    r"""
    Optimize parameters for proatom density functions using a self-consistent (SC) method.

    .. math::

        N_{Ai} = \int \rho_A(r) \frac{\rho_{Ai}^0(r)}{\rho_A^0(r)} dr

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
    logger : logging.Logger or None
        The log object.
    density_cutoff : float
        Density values below this cutoff are considered invalid.
    negative_cutoff : float
        The value less than `negative_cutoff` is treated as negative.
    population_cutoff : float
        The criteria for the difference between the sum of propars and reference population.

    Returns
    -------
    1D np.ndarray
        Optimized proatom parameters.

    Raises
    ------
    RuntimeError
        If the inner iteration does not converge.

    """
    pro_shells, pro, sick, ratio, _ = compute_quantities(
        rho, propars, bs_funcs, density_cutoff, do_ln_ratio=False
    )
    propars[:] = np.einsum("p,ip->i", weights, pro_shells * ratio)
    return propars


def solver_sc_plus_cvxopt(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    logger,
    density_cutoff,
    negative_cutoff,
    population_cutoff,
    # Options for solver.
    sc_iter_limit=1000,
    **cvxopt_options,
):
    r"""
    Optimize parameters for proatom density functions using a mixing of self-consistent (SC) method and convex
    optimization method.

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
    logger : logging.Logger or None
        The log object.
    density_cutoff : float
        Density values below this cutoff are considered invalid.
    negative_cutoff : float
        The value less than `negative_cutoff` is treated as negative.
    population_cutoff : float
        The criteria for the difference between the sum of propars and reference population.
    sc_iter_limit: int
        The number of iteration steps of self-consistent method.
    cvxopt_options: dict
        The options for cvxopt solver. See `solver_cvxopt`.

    Returns
    -------
    1D np.ndarray
        Optimized proatom parameters.

    Raises
    ------
    RuntimeError
        If the inner iteration does not converge.

    """
    oldpro = None
    logger.debug("            Iter.    Change    ")
    logger.debug("            -----    ------    ")
    for irep in range(sc_iter_limit):
        pro_shells, pro, sick, ratio, _ = compute_quantities(
            rho, propars, bs_funcs, density_cutoff, do_ln_ratio=False
        )

        # the partitions and the updated parameters
        propars[:] = np.einsum("p,ip->i", weights, pro_shells * ratio)

        # check for convergence
        if oldpro is None:
            change = 1e100
        else:
            error = oldpro - pro
            change = np.sqrt(np.einsum("i,i,i->", weights, error, error))
        logger.debug(f"            {irep+1:<4}    {change:.3e}")
        if change < threshold:
            pop = np.einsum("i,i", weights, rho)
            check_pro_atom_parameters_non_neg_pars(
                propars,
                logger=logger,
                total_population=float(pop),
                negative_cutoff=negative_cutoff,
                population_cutoff=population_cutoff,
            )
            return propars
        oldpro = pro
    else:
        logger.warning("Inner iteration is not converge! Using LISA-I scheme.")
        cvxopt_options = cvxopt_options or {}
        return solver_cvxopt(
            bs_funcs,
            rho,
            propars=propars,
            points=points,
            weights=weights,
            threshold=threshold,
            logger=logger,
            density_cutoff=density_cutoff,
            negative_cutoff=negative_cutoff,
            **cvxopt_options,
        )


def solver_diis(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    logger,
    density_cutoff,
    negative_cutoff,
    population_cutoff,
    # Options for solver.
    check_mono=True,
    **diis_options,
):
    r"""
    Optimize parameters for proatom density functions using direct inversion in an iterative space (DIIS).

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
    logger : logging.Logger or None
        The log object.
    density_cutoff : float
        Density values below this cutoff are considered invalid.
    negative_cutoff : float
        The value less than `negative_cutoff` is treated as negative.
    population_cutoff : float
        The criteria for the difference between the sum of propars and reference population.
    check_mono: bool, opotionl
        Check if the density is monotonically decaying.
    diis_options: dict
        The other options for the solver.


    Returns
    -------
    1D np.ndarray
        Optimized proatom parameters.

    Raises
    ------
    RuntimeError
        If the inner iteration does not converge.

    """

    def function_g(x):
        """The fixed-point equation :math:`g(x)=x`."""
        pro_shells, _, _, ratio, _ = compute_quantities(
            rho, x, bs_funcs, density_cutoff, do_ln_ratio=False
        )
        return np.einsum("ip,p->i", pro_shells * ratio, weights)

    diis_options = diis_options or {}
    new_propars, niter, history_x = diis(
        propars, function_g, threshold, logger=logger, **diis_options
    )
    pop = np.einsum("i,i", weights, rho)
    check_pro_atom_parameters_neg_pars(
        new_propars,
        bs_funcs,
        logger=logger,
        negative_cutoff=negative_cutoff,
        population_cutoff=population_cutoff,
        total_population=float(pop),
        check_monotonicity=check_mono,
    )
    return new_propars


def solver_newton(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    logger,
    density_cutoff,
    negative_cutoff,
    population_cutoff,
    # Solver options
    **newton_options,
):
    r"""
    Optimize parameters for pro-atom density functions using Newton method

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
    logger : logging.Logger or None
        The log object.
    density_cutoff : float
        Density values below this cutoff are considered invalid.
    negative_cutoff : float
        The value less than `negative_cutoff` is treated as negative.
    population_cutoff : float
        The criteria for the difference between the sum of propars and reference population.
    newton_options: dict
        Other options for the solver.

    Returns
    -------
    1D np.ndarray
        Optimized proatom parameters.

    Raises
    ------
    RuntimeError
        If the inner iteration does not converge.

    """
    return _solver_general_newton(
        bs_funcs,
        rho,
        propars,
        points,
        weights,
        threshold,
        logger,
        density_cutoff,
        negative_cutoff,
        population_cutoff,
        mode="exact",
        **newton_options,
    )


def solver_m_newton(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    logger,
    density_cutoff,
    negative_cutoff,
    population_cutoff,
    # Solver options
    **newton_options,
):
    r"""
    Optimize parameters for pro-atom density functions using Newton method

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
    logger : logging.Logger or None
        The log object.
    density_cutoff : float
        Density values below this cutoff are considered invalid.
    negative_cutoff : float
        The value less than `negative_cutoff` is treated as negative.
    population_cutoff : float
        The criteria for the difference between the sum of propars and reference population.
    newton_options: dict
        Other options for the solver.

    Returns
    -------
    1D np.ndarray
        Optimized proatom parameters.

    Raises
    ------
    RuntimeError
        If the inner iteration does not converge.

    """
    return _solver_general_newton(
        bs_funcs,
        rho,
        propars,
        points,
        weights,
        threshold,
        logger,
        density_cutoff,
        negative_cutoff,
        population_cutoff,
        mode="modified",
        **newton_options,
    )


def solver_quasi_newton(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    logger,
    density_cutoff,
    negative_cutoff,
    population_cutoff,
    # Solver options
    **newton_options,
):
    r"""
    Optimize parameters for pro-atom density functions using Quasi-Newton method

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
    logger : logging.Logger or None
        The log object.
    density_cutoff : float
        Density values below this cutoff are considered invalid.
    negative_cutoff : float
        The value less than `negative_cutoff` is treated as negative.
    population_cutoff : float
        The criteria for the difference between the sum of propars and reference population.
    newton_options: dict
        Other options for the solver.

    Returns
    -------
    1D np.ndarray
        Optimized proatom parameters.

    Raises
    ------
    RuntimeError
        If the inner iteration does not converge.

    """
    return _solver_general_newton(
        bs_funcs,
        rho,
        propars,
        points,
        weights,
        threshold,
        logger,
        density_cutoff,
        negative_cutoff,
        population_cutoff,
        mode="bfgs",
        **newton_options,
    )


def _solver_general_newton(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    logger,
    density_cutoff,
    negative_cutoff,
    population_cutoff,
    # Solver options
    mode="bfgs",
    tau=1.0,
    linspace_size=20,
    check_mono=False,
    niter=10000,
):
    r"""
    Optimize parameters for pro-atom density functions using Quasi-Newton method

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
    logger : logging.Logger or None
        The log object.
    density_cutoff : float
        Density values below this cutoff are considered invalid.
    negative_cutoff : float
        The value less than `negative_cutoff` is treated as negative.
    population_cutoff : float
        The criteria for the difference between the sum of propars and reference population.
    check_mono: bool, opotionl
        Check if the density is monotonically decaying.

    Returns
    -------
    1D np.ndarray
        Optimized proatom parameters.

    Raises
    ------
    RuntimeError
        If the inner iteration does not converge.

    """
    logger.debug("            Iter.    Change    ")
    logger.debug("            -----    ------    ")

    oldpro = None
    change = 1e100

    # if logger.level <= logging.DEBUG:
    #     diff = rho[:-1] - rho[1:]
    #     logger.debug(" Check monotonicity of density ".center(80, "*"))
    #     if len(diff[diff < -1e-5]):
    #         logger.debug(
    #             "The spherical average density is not monotonically decreasing."
    #         )
    #         logger.debug(
    #             "The different between densities on two neighboring points are negative:"
    #         )
    #         logger.debug(diff[diff < -1e-5])
    #     else:
    #         logger.debug("Pass.")

    def linesearch(delta: np.ndarray, propars: np.ndarray):
        """Line search used in Newton-method.

        The 1D density should be monotonically decaying.

        Parameters
        ----------
        delta :
            The Newton step for each propar.
        propars :
            The pro-atom parameters.

        """
        if mode == "exact":
            return propars + delta, delta

        for x in np.linspace(tau, 0, linspace_size):
            _s = x * delta
            new_propars = propars + _s
            _, new_pro, sick, _, _ = compute_quantities(
                rho,
                new_propars,
                bs_funcs,
                density_cutoff,
                do_ratio=False,
                do_ln_ratio=False,
            )
            if check_mono:
                # Check if all elements in new_pro are above density_cutoff
                # and if the values in new_pro are monotonically decreasing
                if (new_pro > negative_cutoff).all() and (
                    new_pro[:-1] - new_pro[1:] > negative_cutoff
                ).all():
                    return new_propars, _s
            else:
                # Check only if all elements in new_pro are above density_cutoff
                if (new_pro > negative_cutoff).all():
                    return new_propars, _s

            # if (new_pro > NEGATIVE_CUTOFF).all() and (
            #     new_pro[:-1] - new_pro[1:] > NEGATIVE_CUTOFF
            # ).all():
            #     return new_propars, _s
        else:
            logger.debug(f"propars: {propars}")
            logger.debug(f"new_propars: {new_propars}")
            raise RuntimeError("Line search failed!")

    H = None
    olddf = None
    oldH = None
    s = None
    pop = np.einsum("i,i", weights, rho)
    for irep in range(niter):
        _, pro, sick, ratio, ln_ratio = compute_quantities(rho, propars, bs_funcs, density_cutoff)
        logger.debug(" Optimized pro-atom parameters:")
        logger.debug(propars)

        # check for convergence
        if oldpro is not None:
            error = oldpro - pro
            change = np.sqrt(np.einsum("i,i,i", weights, error, error))
        logger.debug(f"            {irep+1:<4}    {change:.3e}")
        if change < threshold:
            check_pro_atom_parameters_neg_pars(
                propars,
                total_population=float(pop),
                pro_atom_density=np.asarray(pro),
                logger=logger,
                negative_cutoff=negative_cutoff,
                population_cutoff=population_cutoff,
                check_monotonicity=check_mono,
            )
            return propars

        # update propars
        integrand = bs_funcs * ratio
        with np.errstate(all="ignore"):
            grad_integrand = integrand / pro
        grad_integrand[:, sick] = 0.0

        if mode == "bfgs":
            if irep == 0:
                hess = np.identity(len(propars))
                H = np.linalg.inv(hess)
                df = -np.einsum("kp,p->k", integrand, weights)
            else:
                df = -np.einsum("kp,p->k", integrand, weights)
                H = bfgs(df, s, olddf, oldH)
            delta = H @ (-1 - df)
        elif mode in ["exact", "modified"]:
            df = -np.einsum("kp,p->k", integrand, weights)
            hess = np.einsum("kp, jp, p->kj", grad_integrand, bs_funcs, weights)
            delta = solve(hess, -1 - df, assume_a="sym")
        else:
            raise NotImplementedError

        propars[:], s = linesearch(delta, propars)

        logger.debug(" delta:")
        logger.debug(delta)
        logger.debug("the sum of delta:")
        logger.debug(np.sum(delta))

        oldpro = pro
        olddf = df
        oldH = H
    raise RuntimeError("Inner loop: Newton does not converge!")


def solver_trust_region(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    logger,
    density_cutoff,
    negative_cutoff,
    population_cutoff,
    # Solver options
    explicit_constr=True,
    trust_region_options=None,
):
    """
    Optimize parameters for pro-atom density functions using trust-region method.

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
    logger : logging.Logger or None
        The log object.
    density_cutoff : float
        Density values below this cutoff are considered invalid.
    negative_cutoff : float
        The value less than `negative_cutoff` is treated as negative.
    population_cutoff : float
        The criteria for the difference between the sum of propars and reference population.
    explicit_constr : bool (optional)
        Whether adding an explicit constraint for the atomic population. The default is `False`.
    trust_region_options: dict
        Other options for `trust_region` solver.

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
    pop = np.einsum("i,i", weights, rho)

    constraint = None
    if explicit_constr:
        constraint = LinearConstraint(np.ones((1, nprim)), pop, pop)

    def obj_funcs(x=None):
        """The objective function."""
        pro_shells, pro, sick, ratio, ln_ratio = compute_quantities(
            rho, x, bs_funcs, density_cutoff
        )
        f = np.einsum("i,i,i", weights, rho, ln_ratio)
        df = -np.einsum("j,j,ij->i", weights, ratio, bs_funcs)
        if not explicit_constr:
            f += np.einsum("i,i", weights, pro) - pop
            df += 1
        return f, df

    bounds = [(0.0, 200)] * nprim
    trust_region_options = trust_region_options or {}
    opt_res = minimize(
        obj_funcs,
        x0=propars,
        method="trust-constr",
        jac=True,
        bounds=bounds,
        constraints=constraint,
        # hess="3-point",
        hess=SR1(),
        options={"gtol": threshold * 1e-3, "maxiter": 1000}.update(trust_region_options),
    )

    if not opt_res.success:
        raise RuntimeError("Convergence failure.")

    result = np.asarray(opt_res["x"]).flatten()
    check_pro_atom_parameters_non_neg_pars(
        result,
        bs_funcs,
        total_population=float(pop),
        logger=logger,
        negative_cutoff=negative_cutoff,
        population_cutoff=population_cutoff,
    )
    return result


def solver_cdiis(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    logger,
    density_cutoff,
    negative_cutoff,
    population_cutoff,
    # Solver options
    check_mono=True,
    **cdiis_options,
):
    """
    Optimize parameters for proatom density functions using CDIIS algorithm.


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
    threshold : float, optional
       default value : 1e-08
       tolerence parameter for convergence test on residual (commutator)
    logger : logging.Logger or None
        The log object.
    density_cutoff : float
        Density values below this cutoff are considered invalid.
    negative_cutoff : float
        The value less than `negative_cutoff` is treated as negative.
    population_cutoff : float
        The criteria for the difference between the sum of propars and reference population.
    cdiis_options: dict
        Other options for the solver.
    check_mono: bool, opotionl
        Check if the density is monotonically decaying.

    Returns
    -------
    conv : boolean
       if convergence, True, else, False
    """

    def function_g(x):
        """The objective fixed-point equation."""
        pro_shells, _, _, ratio, _ = compute_quantities(
            rho, x, bs_funcs, density_cutoff, do_ln_ratio=False
        )
        return np.einsum("ip,p->i", pro_shells * ratio, weights)

    conv, nbiter, rnormlist, mklist, cnormlist, xlast, history_x = cdiis(
        propars,
        function_g,
        threshold,
        logger=logger,
        **cdiis_options,
    )
    if not conv:
        raise RuntimeError("Not converged!")
    pop = np.einsum("i,i", weights, rho)
    check_pro_atom_parameters_neg_pars(
        xlast,
        basis_functions=bs_funcs,
        total_population=float(pop),
        logger=logger,
        negative_cutoff=negative_cutoff,
        population_cutoff=population_cutoff,
        check_monotonicity=check_mono,
    )
    return xlast


def setup_bs_helper(part):
    """Setup basis function helper."""
    if part._bs_helper is None:
        if isinstance(part.basis_func, str):
            bs_name = part.basis_func.lower()
            if bs_name in ["gauss", "slater"]:
                part.logger.info(f"Load {bs_name.upper()} basis functions")
                if part.basis_type == "analytic":
                    part._bs_helper = ExpBasisFuncHelper.from_function_type(bs_name)
                elif part.basis_type == "numeric":
                    part._bs_helper = NumericBasisFuncHelper.from_function_type(bs_name)
                else:
                    raise RuntimeError("The bs_type should be one of analytic and numeric.")
            else:
                part.logger.info(f"Load basis functions from custom json file: {part.basis_func}")
                part._bs_helper = ExpBasisFuncHelper.from_file(part.basis_func)
        elif isinstance(part.basis_func, (AnalyticBasisFuncHelper, NumericBasisFuncHelper)):
            part._bs_helper = part.basis_func
        else:
            raise NotImplementedError(
                "The type of basis_func should be one of string or class BasisFuncHelper."
            )
    return part._bs_helper


class LinearISAWPart(GaussianISAWPart):
    r"""
    Implements the Linear Iterative Stockholder Analysis (L-ISA) partitioning scheme.

    This class extends ``GaussianISAWPart`` and specializes in performing
    electron density partitioning in molecules using various L-ISA schemes. L-ISA
    is a method for dividing the electron density of a molecule into atomic
    contributions. This class offers a variety of schemes for this
    partitioning, both at local [1]_ and global optimization levels.

    **Optimization Problem Schemes**

    - Convex optimization (`solver`="cvxopt")
    - Trust-region methods with constraints
        - Implicit constraints (`solver`="trust-constr-im")
        - Explicit constraints (`solver`="trust-constr-ex")
    - Fixed-point methods
        - Alternating/self-consistent method (`solver`="sc")
        - Direct Inversion of Iterative Space (DIIS) (`sovler`="diis")
        - Newton method (`solver`="newton")


    See Also
    --------
    horton_part.lisa_g : Global Linear approximation of ISA.

    References
    ----------
    .. [1] Benda R., et al. Multi-center decomposition of molecular densities: A mathematical perspective.

    """

    name = "lisa"

    builtin_solvers = {
        "cvxopt": solver_cvxopt,
        "sc": solver_sc,
        # For large diis_size, it is also not robust
        "diis": solver_diis,
        # Not robust
        "newton": solver_newton,
        "m-newton": solver_m_newton,
        "quasi-newton": solver_quasi_newton,
        # Explicit constr is faster than implicit
        "trust-region": solver_trust_region,
        "sc-1-iter": solver_sc_1_iter,
        "sc-plus-convex": solver_sc_plus_cvxopt,
        "cdiis": solver_cdiis,
    }

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
        solver="cvxopt",
        solver_options=None,
        basis_func="gauss",
        basis_type="analytic",
        grid_type=1,
        **kwargs,
    ):
        """
        LISA initial function.

        **Optional arguments:** (that are not defined in ``GaussianISAWPart``)

        Parameters
        ----------
        basis_func : string, optional
            The type of basis functions, and Gaussian is the default.

        """
        self.basis_func = basis_func
        if self.basis_func in ["gauss", "slater"]:
            self._func_type = self.basis_func.upper()
        else:
            self._func_type = "Customized"
        self.basis_type = basis_type

        super().__init__(
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            lmax,
            logger,
            threshold,
            maxiter,
            inner_threshold,
            radius_cutoff,
            solver,
            solver_options,
            grid_type,
            **kwargs,
        )

        if self.grid_type not in [1]:
            self.logger.info(
                f"The grid type is {self.grid_type} and please set `check_mono` to `False` when using aLISA+- methods."
            )

    @property
    def bs_helper(self):
        """A basis function helper."""
        return setup_bs_helper(self)

    def _init_log_scheme(self):
        # if log.do_medium:
        info_list = [
            ("Scheme", "Linear Iterative Stockholder"),
            ("Outer loop convergence threshold", "%.1e" % self._threshold),
            (
                "Inner loop convergence threshold",
                "%.1e" % self._inner_threshold,
            ),
            ("Using global ISA", False),
        ]

        info_list.extend(
            [
                ("Maximum outer iterations", self._maxiter),
                ("lmax", self._lmax),
                (
                    "Solver",
                    self._solver.__name__ if callable(self._solver) else self._solver.upper(),
                ),
                ("Basis function type", self._func_type),
                ("Grid type", self.grid_type),
                # ("Local grid radius", self._radius_cutoff),
                # ("Allow negative parameters", allow_negative_params),
            ]
        )
        for k, v in self._solver_options.items():
            info_list.append((f"Solver options -- {k}", str(v)))
        deflist(self.logger, info_list)
        self.logger.info(" ")

    def _opt_propars(
        self,
        bs_funcs,
        rho,
        propars,
        points,
        weights,
        alphas,
        threshold,
    ):
        if callable(self._solver):
            # use customized sovler
            solver = self._solver
        elif self._solver in self.builtin_solvers.keys():
            # use builtin solver
            solver = self.builtin_solvers[self._solver]
        else:
            raise NotImplementedError

        return solver(
            bs_funcs,
            rho,
            propars,
            points,
            weights,
            threshold,
            self.logger,
            density_cutoff=self.density_cutoff,
            negative_cutoff=self.negative_cutoff,
            population_cutoff=self.population_cutoff,
            **self._solver_options,
        )
