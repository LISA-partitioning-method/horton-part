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
"""
Module for Linear Iterative Stockholder Analysis (L-ISA) partitioning scheme.
"""

import logging
import warnings

import cvxopt
import numpy as np
from scipy.linalg import LinAlgWarning, eigh, solve
from scipy.optimize import SR1, LinearConstraint, minimize
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import spsolve

from .core.basis import BasisFuncHelper

# from .core.log import log, biblio
from .core.logging import deflist
from .gisa import GaussianISAWPart
from .utils import check_for_pro_error, check_pro_atom_parameters, compute_quantities

# Suppress specific warning
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


__all__ = [
    "LinearISAWPart",
    "LisaConvexOptWPart",
    "LisaConvexOptNWPart",
    "LisaSelfConsistentWPart",
    "LisaDIISWPart",
    "LisaNewtonWPart",
    "LisaTrustConstraintExpWPart",
    "LisaTrustConstraintImpWPart",
    "opt_propars_fixed_points_sc",
    "opt_propars_fixed_points_diis",
    "opt_propars_fixed_points_newton",
    "opt_propars_minimization_trust_constr",
    "opt_propars_minimization_fast",
]

logger = logging.getLogger(__name__)


class LinearISAWPart(GaussianISAWPart):
    r"""
    Implements the Linear Iterative Stockholder Analysis (L-ISA) partitioning scheme.

    This class extends `GaussianISAWPart` and specializes in performing
    electron density partitioning in molecules using various L-ISA schemes. L-ISA
    is a method for dividing the electron density of a molecule into atomic
    contributions. This class offers a variety of schemes for this
    partitioning, both at local and global optimization levels.

    Optimization Problem Schemes
    ============================

    Local Optimization Problem
    --------------------------

    - Convex optimization (LISA-101)
    - Trust-region methods with constraints
        - Implicit constraints (LSIA-301)
        - Explicit constraints (LSIA-302)
    - Fixed-point methods
        - Alternating/self-consistent method (LISA-201)
        - DIIS (LISA-202, LISA-206)
        - Newton method (LISA-203)

    Global Optimization Problem
    ---------------------------

    - Convex optimization (LISA-101)
    - Trust-region methods with constraints
        - Implicit constraints (LSIA-301)
        - Explicit constraints (LSIA-302)
    - Fixed-point methods
        - Alternating method (LISA-201)
        - DIIS (LISA-202, LISA-206)
        - Newton method (LISA-203)


    See Also
    --------
    horton_part.GaussianISAWPart : Parent class from which this class is derived.

    References
    ----------
    .. [1] TODO

    """

    name = "lisa"

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
        local_grid_radius=np.inf,
        solver=1,
        diis_size=8,
        basis_func_type="gauss",
        basis_func_json_file=None,
    ):
        """
        Construct LISA for given arguments.

        **Optional arguments:** (that are not defined in ``WPart``)

        Parameters
        ----------
        threshold: float
             The procedure is considered to be converged when the maximum
             change of the charges between two iterations drops below this
             threshold.
        maxiter: int
             The maximum number of iterations. If no convergence is reached
             in the end, no warning is given.
             Reduce the CPU cost at the expense of more memory consumption.
        """
        self.func_type = basis_func_type
        self.diis_size = diis_size

        GaussianISAWPart.__init__(
            self,
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            lmax,
            threshold,
            maxiter,
            inner_threshold,
            local_grid_radius,
            solver,
        )

        if basis_func_json_file is not None:
            logger.info(f"Load basis functions from custom json file: {basis_func_json_file}")
            self.bs_helper = BasisFuncHelper.from_json(basis_func_json_file)
        else:
            logger.info(f"Load {basis_func_type} basis functions")
            self.bs_helper = BasisFuncHelper.from_function_type(basis_func_type)

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

        if self._solver in [104, 202, 203, 204, 206]:
            allow_negative_params = True
        else:
            allow_negative_params = False

        info_list.extend(
            [
                ("Maximum outer iterations", self._maxiter),
                ("lmax", self._lmax),
                ("Solver", self._solver),
                ("Basis function type", self.func_type),
                ("Local grid radius", self._local_grid_radius),
                ("Allow negative parameters", allow_negative_params),
            ]
        )
        if self._solver in [202, 206]:
            info_list.append(("DIIS size", self.diis_size))
        deflist(logger, info_list)
        # biblio.cite(
        #     "Benda2022", "the use of Linear Iterative Stockholder partitioning"
        # )

    def _opt_propars(
        self,
        bs_funcs,
        rho,
        propars,
        points,
        weights,
        alphas,
        threshold,
        density_cutoff=1e-15,
    ):
        if self._solver in [1, 101]:
            return opt_propars_minimization_fast(
                bs_funcs, rho, propars, points, weights, threshold, density_cutoff
            )
        if self._solver == 104:
            # no robust: HF, SiH4
            return opt_propars_minimization_fast(
                bs_funcs, rho, propars, points, weights, threshold, density_cutoff, True
            )
        elif self._solver in [2, 201]:
            return opt_propars_fixed_points_sc(
                bs_funcs, rho, propars, points, weights, threshold, density_cutoff
            )
        elif self._solver == 20101:
            return opt_propars_fixed_points_sc_one_step(
                bs_funcs, rho, propars, points, weights, threshold, density_cutoff
            )
        elif self._solver == 2011:
            return opt_propars_fixed_points_sc_convex(
                bs_funcs, rho, propars, points, weights, threshold, density_cutoff
            )
        elif self._solver in [202, 206]:
            # for large diis_size, it is also not robust
            return opt_propars_fixed_points_diis(
                bs_funcs,
                rho,
                propars,
                points,
                weights,
                threshold,
                density_cutoff,
                diis_size=self.diis_size,
            )
        elif self._solver == 203:
            # not robust
            return opt_propars_fixed_points_newton(
                bs_funcs, rho, propars, points, weights, threshold, density_cutoff
            )
        elif self._solver in [3, 301]:
            # same as LISA-102 but with constraint implicitly, slower than 302
            return opt_propars_minimization_trust_constr(
                bs_funcs,
                rho,
                propars,
                points,
                weights,
                threshold,
                density_cutoff,
                explicit_constr=False,
            )
        elif self._solver == 302:
            # use `trust_constr` in SciPy with constraint explicitly
            return opt_propars_minimization_trust_constr(
                bs_funcs,
                rho,
                propars,
                points,
                weights,
                threshold,
                density_cutoff=density_cutoff,
            )
        else:
            raise NotImplementedError


class _LisaOptimizationPart(LinearISAWPart):
    def __init__(self, *args, solver_id, **kwargs):
        kwargs["solver"] = solver_id
        super().__init__(*args, **kwargs)


class LisaConvexOptWPart(_LisaOptimizationPart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, solver_id=101, **kwargs)


class LisaConvexOptNWPart(_LisaOptimizationPart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, solver_id=104, **kwargs)


class LisaSelfConsistentWPart(_LisaOptimizationPart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, solver_id=201, **kwargs)


class LisaNewtonWPart(_LisaOptimizationPart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, solver_id=203, **kwargs)


class LisaDIISWPart(_LisaOptimizationPart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, solver_id=206, **kwargs)


class LisaTrustConstraintImpWPart(_LisaOptimizationPart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, solver_id=301, **kwargs)


class LisaTrustConstraintExpWPart(_LisaOptimizationPart):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, solver_id=302, **kwargs)


def opt_propars_fixed_points_sc(bs_funcs, rho, propars, points, weights, threshold, density_cutoff):
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
    density_cutoff : float
        Density values below this cutoff are considered invalid.

    Returns
    -------
    np.ndarray
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
    oldpro = None
    logger.debug("            Iter.    Change    ")
    logger.debug("            -----    ------    ")
    for irep in range(int(1e10)):
        pro_shells, pro, sick, ratio, lnratio = compute_quantities(
            rho, propars, bs_funcs, density_cutoff
        )

        # the partitions and the updated parameters
        propars[:] = np.einsum("p,ip->i", weights, pro_shells * ratio)
        check_pro_atom_parameters(propars)

        # check for convergence
        if oldpro is None:
            change = 1e100
        else:
            error = oldpro - pro
            change = np.sqrt(np.einsum("i,i,i", weights, error, error))
        logger.debug(f"            {irep+1:<4}    {change:.3e}")
        if change < threshold:
            return propars
        oldpro = pro
    raise RuntimeError("Error: Inner iteration is not converge!")


def opt_propars_fixed_points_sc_one_step(
    bs_funcs, rho, propars, points, weights, threshold, density_cutoff
):
    r"""
    Optimize parameters for proatom density functions using LISA-2 with self-consistent (SC) method.

    The parameters can be computed analytically in this way. which should give the same results
    as the L-ISA algorithms.

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
    pro_shells, pro, sick, ratio, lnratio = compute_quantities(
        rho, propars, bs_funcs, density_cutoff
    )

    # the partitions and the updated parameters
    propars[:] = np.einsum("p,ip->i", weights, pro_shells * ratio)
    return propars


def opt_propars_fixed_points_sc_convex(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    density_cutoff,
    sc_iter_limit=1000,
):
    r"""
    Optimize parameters for proatom density functions using LISA-2 with self-consistent (SC) method.

    The parameters can be computed analytically in this way. which should give the same results
    as the L-ISA algorithms.

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
    oldpro = None
    # if log.do_medium:
    logger.debug("            Iter.    Change    ")
    logger.debug("            -----    ------    ")
    for irep in range(sc_iter_limit):
        pro_shells, pro, sick, ratio, lnratio = compute_quantities(
            rho, propars, bs_funcs, density_cutoff
        )

        # the partitions and the updated parameters
        propars[:] = np.einsum("p,ip->i", weights, pro_shells * ratio)
        check_pro_atom_parameters(propars)

        # check for convergence
        if oldpro is None:
            change = 1e100
        else:
            error = oldpro - pro
            change = np.sqrt(np.einsum("i,i,i->", weights, error, error))
        # if log.do_medium:
        #     log(f"            {irep+1:<4}    {change:.3e}")
        if change < threshold:
            return propars
        oldpro = pro
    else:
        warnings.warn("Inner iteration is not converge! Using LISA-I scheme.")
        return opt_propars_minimization_fast(
            bs_funcs, rho, propars, points, weights, threshold, density_cutoff
        )
    # raise RuntimeError("Error: Inner iteration is not converge!")


def diis(c_values, r_values, max_history):
    """
    c_values: List of recent iterates
    r_values: List of recent residuals/errors
    max_history: Maximum number of history values to consider
    """

    # Limit history
    if len(c_values) > max_history:
        c_values = c_values[-max_history:]
        r_values = r_values[-max_history:]

    n = len(c_values)

    # Build the B matrix
    B = np.zeros((n + 1, n + 1))
    B[-1, :] = -1
    B[:, -1] = -1
    B[-1, -1] = 0

    for i in range(n):
        for j in range(n):
            B[i, j] = np.dot(r_values[i], r_values[j])

    turn_off_diis = False
    # Right-hand side vector
    rhs = np.zeros(n + 1)
    rhs[-1] = -1

    # Solve for the coefficients
    w, v = eigh(B)
    tol = 1e-14
    if np.any(abs(w) < tol):
        warnings.warn("Linear dependence found in DIIS error vectors.")
        idx = abs(w) > tol
        coeffs = np.dot(v[:, idx] * (1.0 / w[idx]), np.dot(v[:, idx].T.conj(), rhs))
    else:
        try:
            coeffs = np.linalg.solve(B, rhs)
        except np.linalg.linalg.LinAlgError as e:
            warnings.warn(" diis singular, eigh(h) %s", w)
            raise e
    # warnings.warn("diis-c %s", coeffs)

    turn_off_diis = False
    nb_coeff = len(coeffs[:-1])
    if nb_coeff > 2 and np.allclose(coeffs[:-1], np.ones_like(coeffs[:-1]) / nb_coeff, rtol=1e-3):
        turn_off_diis = True
        logger.info("turn off DIIS")

    # assert np.isclose(np.sum(coeffs[:-1]), 1.0)
    # B_rank = np.linalg.matrix_rank(B)
    # if B_rank != B.shape[0]:
    #     print(B_rank, B.shape)

    # Build the DIIS solution
    c_new = np.zeros_like(c_values[0])
    for i in range(n):
        c_new += coeffs[i] * c_values[i]

    return c_new, turn_off_diis


def opt_propars_fixed_points_diis(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    density_cutoff,
    diis_size=8,
):
    r"""
    Optimize parameters for proatom density functions using MBIS Lagrange.

    The parameters can be computed analytically in this way. which should give the same results
    as the L-ISA algorithms.

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
    history_diis = []
    history_propars = []
    start_diis_iter = diis_size + 1

    # if log.do_medium:
    logger.debug("            Iter.    dRMS      ")
    logger.debug("            -----    ------    ")

    for irep in range(1000):
        pro_shells, pro, sick, ratio, lnratio = compute_quantities(
            rho, propars, bs_funcs, density_cutoff
        )

        integrands = pro_shells * ratio
        fun_val = np.einsum("ip,p->i", integrands, weights)

        # Build DIIS Residual
        diis_r = propars - fun_val
        # Compute drms
        drms = np.linalg.norm(diis_r)

        # if log.do_medium:
        logger.debug(f"           {irep:<4}    {drms:.6E}")
        if drms < threshold:
            return propars

        # Append trail & residual vectors to lists
        if irep >= start_diis_iter - diis_size:
            history_propars.append(fun_val)
            history_diis.append(diis_r)

        if irep >= start_diis_iter:
            # Build B matrix
            propars_prev = history_propars[-diis_size:]
            diis_prev = history_diis[-diis_size:]

            B_dim = len(propars_prev) + 1
            B = np.zeros((B_dim, B_dim))
            B[-1, :] = B[:, -1] = -1
            tmp = np.einsum("ip,jp->ij", diis_prev, diis_prev)
            B[:-1, :-1] = (tmp + tmp.T) / 2
            B[-1, -1] = 0

            # Build RHS of Pulay equation
            rhs = np.zeros(B_dim)
            rhs[-1] = -1
            coeff = spsolve(B, rhs)
            propars = np.einsum("i, ip->p", coeff[:-1], np.asarray(propars_prev))
            if (np.isnan(propars)).any():
                propars = fun_val
        else:
            propars = fun_val

        check_pro_atom_parameters(propars)

    raise RuntimeError("Error: inner iteration is not converge!")


def opt_propars_fixed_points_newton(
    bs_funcs, rho, propars, points, weights, threshold, density_cutoff
):
    r"""
    Optimize parameters for proatom density functions using MBIS Lagrange.

    The parameters can be computed analytically in this way. which should give the same results
    as the L-ISA algorithms.

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
    logger.debug("            Iter.    Change    ")
    logger.debug("            -----    ------    ")

    oldpro = None
    change = 1e100

    if logger.level <= logging.DEBUG:
        diff = rho[:-1] - rho[1:]
        logger.debug(" Check monotonicity of density ".center(80, "*"))
        if len(diff[diff < -1e-5]):
            logger.debug("The spherical average density is not monotonically decreasing.")
            logger.debug("The different between densities on two neighboring points are negative:")
            logger.debug(diff[diff < -1e-5])
        else:
            logger.debug("Pass.")

    for irep in range(1000):
        _, pro, sick, ratio, _ = compute_quantities(rho, propars, bs_funcs, density_cutoff)
        integrand = bs_funcs * ratio
        logger.debug(" Optimized pro-atom parameters:")
        logger.debug(propars)

        # check for convergence
        if oldpro is not None:
            error = oldpro - pro
            change = np.sqrt(np.einsum("i,i,i", weights, error, error))
        logger.debug(f"            {irep+1:<4}    {change:.3e}")
        if change < threshold:
            # check_pro_atom_parameters(
            #     propars, total_population=float(np.sum(pro)), pro_atom_density=pro
            # )
            return propars

        # update propars
        with np.errstate(all="ignore"):
            grad_integrand = integrand / pro
        grad_integrand[:, sick] = 0.0

        jacob = np.einsum("kp, jp, p->kj", grad_integrand, bs_funcs, weights)
        h = 1 - np.einsum("kp,p->k", integrand, weights)
        delta = solve(jacob, -h, assume_a="sym")
        logger.debug(" delta:")
        logger.debug(delta)
        propars += delta
        oldpro = pro
    raise RuntimeError("Inner loop: Newton does not converge!")


def opt_propars_minimization_fast(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    density_cutoff,
    allow_neg_params=False,
):
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

        if logger.level >= logging.DEBUG:
            # if log.do_debug:
            try:
                check_for_pro_error(pro)
            except RuntimeError as e:
                logger.info(bs_funcs.shape)
                logger.info("bs_funcs.T:")
                logger.info(bs_funcs.T)
                logger.info("[in obj_func]: pro-atom parameters:")
                logger.info(np.asarray(x))
                logger.info("pro-atom density:")
                logger.info(pro.reshape((-1, 1)))
                logger.info("atomic density:")
                logger.info(rho.reshape((-1, 1)))
                np.savez_compressed(
                    "dump_negative_dens.npz",
                    points=points,
                    weights=weights,
                    bs_funcs=bs_funcs,
                    propars=np.asarray(x),
                    pro=pro,
                    rho=rho,
                )
                raise e

        # f = -np.einsum("i,i,i", int_weights, rho, ln_pro)
        f = np.einsum("i,i,i", weights, rho, ln_ratio)

        # compute gradient
        grad = weights * ratio
        # if log.do_debug:
        #     check_for_grad_error(grad)
        df = -np.einsum("j,ij->i", grad, bs_funcs)
        df = cvxopt.matrix(df.reshape((1, nprim)))
        if z is None:
            return f, df

        # compute hessian
        hess = np.divide(grad, pro, out=np.zeros_like(grad), where=~sick)
        d2f = np.einsum("k,ik,jk->ij", hess, bs_funcs, bs_funcs)
        # if log.do_debug:
        #     check_for_hessian_error(d2f)
        d2f = z[0] * cvxopt.matrix(d2f)
        return f, df, d2f

    opt_CVX = cvxopt.solvers.cp(
        obj_func,
        G=matrix_constraint_ineq,
        h=vector_constraint_ineq,
        A=matrix_constraint_eq,
        b=vector_constraint_eq,
        options={"show_progress": 3, "feastol": threshold},
        # options={"show_progress": log.do_medium, "feastol": threshold},
    )

    optimized_res = opt_CVX["x"]
    new_propars = np.asarray(optimized_res).flatten()
    check_pro_atom_parameters(new_propars, bs_funcs)
    return new_propars


def opt_propars_minimization_trust_constr(
    bs_funcs,
    rho,
    propars,
    points,
    weights,
    threshold,
    density_cutoff,
    explicit_constr=True,
):
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
    opt_res = minimize(
        obj_funcs,
        x0=propars,
        method="trust-constr",
        jac=True,
        bounds=bounds,
        constraints=constraint,
        # hess="3-point",
        hess=SR1(),
        options={"gtol": threshold * 1e-3, "maxiter": 1000},
    )

    if not opt_res.success:
        raise RuntimeError("Convergence failure.")

    optimized_res = opt_res["x"]
    new_propars = np.asarray(optimized_res).flatten()
    check_pro_atom_parameters(new_propars, bs_funcs, total_population=float(pop))
    return new_propars
