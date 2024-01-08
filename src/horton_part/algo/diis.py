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
"""
Module of CDIIS algorithm.
"""

import logging
import warnings

import numpy as np
import qpsolvers
from scipy.linalg import eigh
from scipy.sparse.linalg import spsolve

__all__ = [
    "diis",
    "lstsq_spsolver",
    "lstsq_solver_dyn",
    "lstsq_solver_with_extra_constr",
]

logger = logging.getLogger(__name__)


def diis(
    x0,
    func,
    threshold,
    maxiter=1000,
    diis_size=8,
    version="P",
    lstsq_solver=None,
):
    r"""
    DIIS algorithm.

    Parameters
    ----------
    x0 : array_like
        1D array, initial values.
    func : callable
        Function `g` which satisfies :math:`g(x)=x`
    threshold : float
        Convergence threshold for the iterative process.
    threshold : float, optional
       default value : 1e-08
       tolerence parameter for convergence test on residual (commutator)
    maxiter : integer, optional
       default value : 50
       maximal number of iterations allowed
    version : str, optional
        The version of DIIS formalism. [1]_
    diis_size : integer, optional
       default value : 5
       size of the window of stored previous iterates in the FD-CDIIS algorithm
       this dimension is also used for the adaptative algorithm
    lstsq_solver : str, optional
        The different method for least-square problem to determine combination coefficients.


    Returns
    -------
    np.array
        The optimized parameters.

    References
    ----------
    [1]: TODO.

    """
    lstsq_solver = lstsq_solver or lstsq_spsolver
    history_r = []
    history_x = []
    history_g = []

    logger.debug("            Iter.    dRMS      ")
    logger.debug("            -----    ------    ")

    x = x0
    for i in range(maxiter):
        g_i = func(x)  # propars = x_i
        r_i = g_i - x
        drms = np.linalg.norm(r_i)
        logger.debug(f"           {i:<4}    {drms:.6E}")
        if drms < threshold:
            return x

        # Append trail & residual vectors to lists
        history_r.append(r_i)
        depth = min(i + 1, diis_size)
        r_list = history_r[-depth:]

        if version == "P":
            # Anderson-Pulay version P
            history_x.append(x.copy())
            x_list = history_x[-depth:]
            x_tilde = lstsq_solver(x_list, r_list)
            x = func(x_tilde)
        else:
            # Anderson-Pulay version A
            history_g.append(g_i)
            g_list = history_g[-depth:]
            x = lstsq_solver(g_list, r_list)
    raise RuntimeError("Error: not converge!")


def lstsq_spsolver(propars_list, residues_list):
    """Solving least-square problem using `spsolver`.

    Parameters
    ----------
    propars_list : array_like
        A list of pro-atom parameters.
    residues_list : array_like
        A list of residues.

    Returns
    -------
    np.ndarray
        1D array, pro-atom parameters.

    """
    mat_size = len(propars_list) + 1
    B = np.zeros((mat_size, mat_size))
    r2 = np.einsum("ip,jp->ij", residues_list, residues_list)
    B[:-1, :-1] = (r2 + r2.T) / 2
    B[-1, :-1] = B[:-1, -1] = -1
    rhs = np.zeros(mat_size)
    rhs[-1] = -1
    sol = spsolve(B, rhs)
    result = np.einsum("i, ip->p", sol[:-1], np.asarray(propars_list))
    if (np.isnan(result)).any():
        logger.info("DIIS: singular matrix.")
        result = propars_list[-1]
    return result


def lstsq_solver_dyn(propars_list, residues_list):
    """Solving least-square problem with dynamic subspace size.

    Parameters
    ----------
    propars_list : array_like
        A list of pro-atom parameters.
    residues_list : array_like
        A list of residues.

    Returns
    -------
    np.ndarray
        1D array, pro-atom parameters.

    """
    mat_size = len(propars_list) + 1
    if mat_size == 2:
        return propars_list[-1]

    B = np.zeros((mat_size, mat_size))
    r2 = np.einsum("ip,jp->ij", residues_list, residues_list)
    B[:-1, :-1] = (r2 + r2.T) / 2
    B[-1, :-1] = B[:-1, -1] = -1
    rhs = np.zeros(mat_size)
    rhs[-1] = -1

    # Solve for the coefficients
    tol = 1e-14
    begin = 0
    while begin < mat_size - 1:
        # TODO: this is problematic because eigh assume B is positive-definite.
        w, v = eigh(B[begin:, begin:])
        nb_small_val = np.sum(abs(w) < tol)
        if not nb_small_val:
            sol = (v * 1 / w) @ (v.T @ rhs[begin:])
            result = np.einsum("i, ip->p", sol[:-1], np.asarray(propars_list[begin:]))
            if (result < -1e-8).any():
                # result = propars_list[-1]
                logger.warning(
                    "Use result from the last iteration due to negative parameters found!"
                )
            logger.debug(f"Updated size of DIIS subspace: {len(sol[:-1])}")
            break
        else:
            begin += nb_small_val
    else:
        warnings.warn("Linear dependence found in DIIS error vectors.")
        result = propars_list[-1]
        logger.debug("real DIIS size: 1")
    return result


def lstsq_solver_with_extra_constr(propars_list, residues_list):
    """
    Solving least-square problem with non-negative parameters constraints.

    Parameters
    ----------
    propars_list : array_like
        A list of pro-atom parameters.
    residues_list : array_like
        A list of residues.

    Returns
    -------
    np.ndarray
        1D array with shape = (N, ) where `N` is the number pro-atom parameters.

    """
    # Build B matrix
    npar = len(propars_list[0])
    space_size = len(propars_list)
    P = np.einsum("ip,jp->ij", residues_list, residues_list)
    # Note: P could be singular.
    P = 0.5 * (P + P.T)
    q = np.zeros((space_size, 1), float)
    # Linear inequality constraints
    G = -np.asarray(propars_list).T
    assert G.shape == (npar, space_size)
    h = np.zeros((1, npar))
    # Linear equality constraints
    A = np.ones((1, space_size))
    b = np.ones((1, 1))

    sol = qpsolvers.solve_qp(P, q, G, h, A, b, solver="osqp", eps_rel=1e-5, eps_abs=1e-3)
    if sol is None:
        raise RuntimeError("No solution fond")
    result = np.einsum("i, ip->p", sol, np.asarray(propars_list))
    return result
