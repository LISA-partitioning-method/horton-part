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


from __future__ import division, print_function
import numpy as np
import cvxopt
from scipy.linalg import solve, eigh, LinAlgWarning, LinAlgError
from scipy.optimize import minimize, LinearConstraint, fsolve, fixed_point
import warnings
from .log import log, biblio
from .gisa import GaussianIterativeStockholderWPart, get_gauss_function

# Suppress specific warning
warnings.filterwarnings("ignore", category=LinAlgWarning)


__all__ = ["LinearIterativeStockholderWPart"]


class LinearIterativeStockholderWPart(GaussianIterativeStockholderWPart):
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
        solver=1,
        diis_size=8,
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
        self.diis_size = diis_size
        GaussianIterativeStockholderWPart.__init__(
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
            solver,
        )

    def _init_log_scheme(self):
        if log.do_medium:
            info_list = [
                ("Scheme", "Linear Iterative Stockholder"),
                ("Outer loop convergence threshold", "%.1e" % self._threshold),
                (
                    "Inner loop convergence threshold",
                    "%.1e" % self._inner_threshold,
                ),
                ("Maximum iterations", self._maxiter),
                ("lmax", self._lmax),
                ("Solver", self._solver),
            ]
            if self._solver in [202]:
                info_list.append(("DIIS size", self.diis_size))
            log.deflist(info_list)
            biblio.cite(
                "Benda2022", "the use of Linear Iterative Stockholder partitioning"
            )

    def _opt_propars(self, rho, propars, rgrid, alphas, threshold):
        if self._solver == 1:
            return _opt_propars_with_lisa_method_fast(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._solver == 101:
            # code optimization of LISA-1
            return _opt_propars_with_lisa_method_slow(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._solver == 102:
            # use `trust_constr` in SciPy with constraint explicitly
            return _opt_propars_with_lisa_method_scipy(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._solver == 103:
            # same as LISA-102 but with constraint implicitly
            return _opt_propars_with_lisa_method_scipy_no_constr(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._solver == 2:
            return _opt_propars_with_mbis_lagrangian(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._solver == 21:
            return _opt_propars_with_mbis_lagrangian_with_lisa(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._solver == 201:
            return _opt_propars_with_mbis_lagrangian_damping(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._solver == 202:
            return _opt_propars_with_lagrangian_diis(
                rho, propars, rgrid, alphas, threshold, diis_size=self.diis_size
            )
        elif self._solver == 203:
            return _opt_propars_with_lagrangian_newton(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._solver == 204:
            return _opt_propars_with_non_linear_equations(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._solver == 205:
            return _opt_propars_with_fix_points(rho, propars, rgrid, alphas, threshold)
        elif self._solver == 206:
            return _opt_propars_with_lagrangian_diis_2(
                rho, propars, rgrid, alphas, threshold, diis_size=self.diis_size
            )
        elif self._solver == 207:
            return _opt_propars_with_fix_point_trust_constr(
                rho, propars, rgrid, alphas, threshold
            )
        # elif str(self._solver).startswith("202"):
        #     solver = str(self._solver)
        #     diis_size = solver.lstrip("202")
        #     if len(diis_size) == 0:
        #         diis_size = 8
        #     else:
        #         diis_size = int(diis_size)
        #     return _opt_propars_with_lagrangian_diis(
        #         rho, propars, rgrid, alphas, threshold, diis_size=diis_size
        #     )
        elif self._solver == 0:
            return _solver_comparison(rho, propars, rgrid, alphas, threshold)
        else:
            raise NotImplementedError


def _opt_propars_with_mbis_lagrangian(rho, propars, rgrid, alphas, threshold):
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
    nprim = len(propars)
    r = rgrid.points
    # avoid too large r
    r = np.clip(r, 1e-100, 1e10)
    oldpro = None
    if log.do_medium:
        log("            Iter.    Change    ")
        log("            -----    ------    ")
    for irep in range(int(1e10)):
        # compute the contributions to the pro-atom
        terms = np.array(
            [get_gauss_function(propars[k], alphas[k], r) for k in range(nprim)]
        )
        if not np.all(terms >= 0.0).all():
            raise RuntimeError("Error: negative pro-shell density found!")
        pro = terms.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)
        # transform to partitions
        terms *= rho / pro
        # the partitions and the updated parameters
        for k in range(nprim):
            propars[k] = rgrid.integrate(4 * np.pi * r**2, terms[k])
        # check for convergence
        if oldpro is None:
            change = 1e100
        else:
            error = oldpro - pro
            change = np.sqrt(rgrid.integrate(4 * np.pi * r**2, error, error))
        if log.do_medium:
            log(f"            {irep+1:<4}    {change:.3e}")
        if change < threshold:
            return propars
        oldpro = pro
    raise RuntimeError("Error: Inner iteration is not converge!")


def _opt_propars_with_mbis_lagrangian_with_lisa(rho, propars, rgrid, alphas, threshold):
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
    oldpropars = propars.copy()
    nprim = len(propars)
    r = rgrid.points
    # avoid too large r
    r = np.clip(r, 1e-100, 1e10)
    oldpro = None
    if log.do_medium:
        log("            Iter.    Change    ")
        log("            -----    ------    ")
    for irep in range(1000):
        # compute the contributions to the pro-atom
        terms = np.array(
            [get_gauss_function(propars[k], alphas[k], r) for k in range(nprim)]
        )
        pro = terms.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)
        # transform to partitions
        terms *= rho / pro
        # the partitions and the updated parameters
        for k in range(nprim):
            propars[k] = rgrid.integrate(4 * np.pi * r**2, terms[k])
        # check for convergence
        if oldpro is None:
            change = 1e100
        else:
            error = oldpro - pro
            change = np.sqrt(rgrid.integrate(4 * np.pi * r**2, error, error))
        if log.do_medium:
            log(f"            {irep+1:<4}    {change:.3e}")
        if change < threshold:
            return propars
        oldpro = pro

    print("Inner iteration is not converge, run lisa-1!")
    new_propars = _opt_propars_with_lisa_method_fast(
        rho, oldpropars, rgrid, alphas, threshold
    )
    return new_propars


def _opt_propars_with_mbis_lagrangian_damping(rho, propars, rgrid, alphas, threshold):
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
    nprim = len(propars)
    r = rgrid.points
    # avoid too large r
    r = np.clip(r, 1e-100, 1e10)
    oldpro = None
    oldprapars = propars.copy()
    if log.do_medium:
        log("            Iter.    Change    ")
        log("            -----    ------    ")
    for irep in range(int(1e10)):
        # compute the contributions to the pro-atom
        terms = np.array(
            [get_gauss_function(propars[k], alphas[k], r) for k in range(nprim)]
        )
        pro = terms.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)
        # transform to partitions
        terms *= rho / pro
        # the partitions and the updated parameters
        for k in range(nprim):
            propars[k] = rgrid.integrate(4 * np.pi * r**2, terms[k])
        propars = propars + 0.9 * (-propars + oldprapars)
        # check for convergence
        if oldpro is None:
            change = 1e100
        else:
            error = oldpro - pro
            change = np.sqrt(rgrid.integrate(4 * np.pi * r**2, error, error))

        if log.do_medium:
            log(f"            {irep+1:<4}    {change:.3e}")

        if change < threshold:
            return propars
        oldpro = pro
    raise RuntimeError("Inner iteration is not converge!")


def _opt_propars_with_lagrangian_diis(
    rho, propars, rgrid, alphas, threshold, diis_size=8
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
    nprim = len(propars)
    r = rgrid.points
    weights = rgrid.weights
    # avoid too large r
    r = np.clip(r, 1e-100, 1e10)
    bs_funcs = np.array([get_gauss_function(1.0, alphas[k], r) for k in range(nprim)])

    history_diis = []
    history_pros = []
    history_shells = []
    start_diis_iter = diis_size

    oldpro = None
    if log.do_medium:
        log("            Iter.    dRMS      ")
        log("            -----    ------    ")

    turn_off_diis = False
    for irep in range(1000):
        # compute the contributions to the pro-atom
        shells = propars[:, None] * bs_funcs
        pro = shells.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)

        # Build DIIS Residual
        diis_r = pro if irep == 0 else pro - oldpro

        # Append trail & residual vectors to lists
        if irep >= start_diis_iter - diis_size:
            history_shells.append(shells)
            history_pros.append(pro)
            history_diis.append(diis_r)

        # Compute drms
        drms = np.sqrt(rgrid.integrate(4 * np.pi * r**2, diis_r, diis_r))
        if log.do_medium:
            log(f"           {irep:<4}    {drms:.6E}")

        if drms < threshold:
            # check N_a
            N_a = rgrid.integrate(4 * np.pi * r**2, rho)
            assert np.isclose(N_a, np.sum(propars), rtol=1e-8)
            if not np.all(propars >= -1e-10).all():
                raise RuntimeError("Error: negative propars found!")
            return propars

        if not turn_off_diis and irep >= start_diis_iter:
            # Build B matrix
            shells_prev = history_shells[-diis_size:]
            pros_prev = history_pros[-diis_size:]
            diss_prev = history_diis[-diis_size:]

            B_dim = len(pros_prev) + 1
            B = np.zeros((B_dim, B_dim))
            B[-1, :] = B[:, -1] = -1
            B[:-1, :-1] = np.einsum("ip,jp->ij", diss_prev, diss_prev)
            B[-1, -1] = 0

            # Build RHS of Pulay equation
            rhs = np.zeros(B_dim)
            rhs[-1] = -1

            # Solve Pulay equation for coeff with numpy
            # Use solve from Scipy, which prints warning info
            try:
                w, v = eigh(B)
                if np.any(abs(w) < 1e-30):
                    warnings.warn(
                        "Linear dependence found in DIIS error vectors. Turn off DIIS"
                    )
                    turn_off_diis = True
                    # return to LISA-2, use purely self-consistent method.
                else:
                    coeff = solve(B, rhs, assume_a="sym")
                    if log.do_debug:
                        assert np.isclose(np.sum(coeff[:-1]), 1)

                    # Build DIIS pro and shells
                    shells = np.einsum(
                        "i, inp->np", coeff[:-1], np.asarray(shells_prev)
                    )
                    pro = np.einsum("i, ip->p", coeff[:-1], np.asarray(pros_prev))
                    # if not np.all(shells >= -1e-10):
                    #     raise RuntimeError("Error: negative pro-shell density found!")
                    # The pro is linear combination of the previous pro-atoms densities, so if the coefficients are not
                    # correct then the pro is invalid.
                    if log.do_debug and not np.all(pro >= -1e-10):
                        warnings.warn("Negative pro-atom density found!")
                    # clip negative part
                    shells = np.clip(shells, 1e-100, np.inf)
                    pro = np.clip(pro, 1e-100, np.inf)
            except LinAlgError:
                turn_off_diis = True

        # compute new poppars from DIIS pro or fix-point
        propars[:] = np.einsum(
            "ip,p->i", shells * rho / pro, 4 * np.pi * r**2 * weights
        )
        oldpro = pro

        if log.do_debug:
            print("propars:", propars)

    raise RuntimeError("Error: inner iteration is not converge!")


def _opt_propars_with_lagrangian_diis_2(
    rho, propars, rgrid, alphas, threshold, diis_size=8
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
    nprim = len(propars)
    r = rgrid.points
    weights = rgrid.weights
    # avoid too large r
    r = np.clip(r, 1e-100, 1e10)
    bs_funcs = np.array([get_gauss_function(1.0, alphas[k], r) for k in range(nprim)])

    history_diis = []
    history_propars = []
    start_diis_iter = diis_size + 1

    if log.do_medium:
        log("            Iter.    dRMS      ")
        log("            -----    ------    ")

    for irep in range(1000):
        # compute the contributions to the pro-atom
        shells = propars[:, None] * bs_funcs
        pro = shells.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)
        integrands = shells * rho / pro
        fun_val = np.einsum("ip,p->i", integrands, 4 * np.pi * r**2 * weights)

        # Build DIIS Residual
        diis_r = propars - fun_val
        # Compute drms
        drms = np.linalg.norm(diis_r)

        if log.do_medium:
            log(f"           {irep:<4}    {drms:.6E}")
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

            coeff = solve(B, rhs)
            # Use lstqs solver
            # coeff = lstsq(B, rhs, cond=1e-14)[0]
            assert np.isclose(np.sum(coeff[:-1]), 1)
            propars = np.einsum("i, ip->p", coeff[:-1], np.asarray(propars_prev))
        else:
            propars = fun_val

        if log.do_debug:
            print("propars:", propars)
        if not np.all(propars >= -1e-10).all():
            raise RuntimeError("Error: negative c_ak found!")

    raise RuntimeError("Error: inner iteration is not converge!")


def _opt_propars_with_fix_points(rho, propars, rgrid, alphas, threshold):
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
    nprim = len(propars)
    r = rgrid.points
    weights = rgrid.weights
    # avoid too large r
    r = np.clip(r, 1e-100, 1e10)
    bs_funcs = np.array([get_gauss_function(1.0, alphas[k], r) for k in range(nprim)])

    def f(x0):
        shells = x0[:, None] * bs_funcs
        pro = shells.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)
        integrands = shells * rho / pro
        return np.einsum("ip,p->i", integrands, 4 * np.pi * r**2 * weights)

    new_propars = fixed_point(
        f, propars, maxiter=int(1e10), method="iteration", xtol=threshold
    )
    # new_propars = fixed_point(f, propars, maxiter=10000, xtol=1e-6)
    # print("propars", new_propars)
    return new_propars


def _opt_propars_with_lagrangian_newton(rho, propars, rgrid, alphas, threshold):
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
    nprim = len(propars)
    r = rgrid.points
    # avoid too large r
    r = np.clip(r, 1e-100, 1e10)
    weights = rgrid.weights
    bs_funcs = np.array([get_gauss_function(1.0, alphas[k], r) for k in range(nprim)])
    int_weights = 4 * np.pi * r**2 * weights

    if log.do_medium:
        log("            Iter.    Change    ")
        log("            -----    ------    ")

    oldpro = None
    change = 1e100
    for irep in range(1000):
        # compute the contributions to the pro-atom
        shells = propars[:, None] * bs_funcs
        if not np.all(shells >= -1e-5).all():
            raise RuntimeError("Error: negative pro-shell density found!")
        pro = shells.sum(axis=0)
        if not np.all(pro >= -1e-10).all():
            raise RuntimeError("Error: negative pro-atom density found!")
        pro = np.clip(pro, 1e-100, np.inf)
        integrand = bs_funcs * rho / pro

        # check for convergence
        if oldpro is not None:
            error = oldpro - pro
            change = np.sqrt(rgrid.integrate(4 * np.pi * r**2, error, error))
        if log.do_medium:
            log(f"            {irep+1:<4}    {change:.3e}")
        if change < threshold:
            return propars

        # update propars
        grad = np.einsum("kp, jp, p->kj", integrand / pro, bs_funcs, int_weights)
        h = 1 - np.einsum("kp,p->k", integrand, int_weights)
        delta = solve(grad, -h, assume_a="sym")
        propars += delta
        oldpro = pro
        if not np.all(propars >= -1e-10):
            raise RuntimeError("Error: negative propars found!")
    raise RuntimeError("Inner loop: Newton does not converge!")


def _opt_propars_with_non_linear_equations(rho, propars, rgrid, alphas, threshold):
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
    nprim = len(propars)
    r = rgrid.points
    # avoid too large r
    r = np.clip(r, 1e-100, 1e10)
    weights = rgrid.weights
    bs_funcs = np.array([get_gauss_function(1.0, alphas[k], r) for k in range(nprim)])
    int_weights = 4 * np.pi * r**2 * weights

    def func(vars):
        shells = vars[:, None] * bs_funcs
        pro = shells.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)
        return 1 - np.einsum("kp,p->k", bs_funcs * rho / pro, int_weights)

    def fprime(vars):
        shells = vars[:, None] * bs_funcs
        pro = shells.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)
        integrand = bs_funcs * rho / pro
        grad = np.einsum("kp, jp, p->kj", integrand / pro, bs_funcs, int_weights)
        return grad

    # TODO: xtol is relative error not absolute
    solution, infodict, iter, msg = fsolve(
        func, propars, fprime=fprime, xtol=threshold, maxfev=1000, full_output=True
    )
    if log.do_medium:
        print(f"iter: {iter}")
        print(msg)

    # without fprime, it will get in trouble for some atoms, e.g., HF.
    # solution = fsolve(func, propars, xtol=threshold)

    if not np.all(solution >= -1e-10):
        print("New propars", solution)
        raise RuntimeError("Error: negative propars found!")
    else:
        return solution


def _opt_propars_with_fix_point_trust_constr(
    rho, propars, rgrid, alphas, threshold, verbose=False
):
    # TODO: not robust
    nprim = len(propars)
    r = rgrid.points
    # avoid too large r
    r = np.clip(r, 1e-100, 1e10)
    weights = rgrid.weights
    bs_funcs = np.array([get_gauss_function(1.0, alphas[k], r) for k in range(nprim)])
    int_weights = 4 * np.pi * r**2 * weights

    def F(vars):
        shells = vars[:, None] * bs_funcs
        pro = shells.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)
        c_expr = 1 - np.einsum("kp,p->k", bs_funcs * rho / pro, int_weights)
        func = np.linalg.norm(c_expr) ** 2

        c_grad = np.einsum(
            "kp, jp, p->kj", bs_funcs * rho / pro**2, bs_funcs, int_weights
        )
        grad = 2 * c_grad @ c_expr
        return func, grad

    bounds = [(-1e-10, 1000)] * nprim
    opt_res = minimize(
        F,
        x0=propars,
        method="trust-constr",
        jac=True,
        bounds=bounds,
        constraints=None,
        hess="3-point",
        options={"gtol": 1e-12, "maxiter": 50000},
    )

    if not opt_res.success:
        raise RuntimeError("Convergence failure.")

    optimized_res = opt_res["x"]
    if not (np.asarray(optimized_res) > 0).all() and log.do_warning:
        log("Not all values are positive!")

    return optimized_res


def _opt_propars_with_lisa_method_slow(
    rho, propars, rgrid, alphas, threshold, verbose=False
):
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
    # Conversion of the identity matrix into CVXOPT format :
    # G = matrix_constraint_ineq
    nprim = len(propars)
    matrix_constraint_ineq = -cvxopt.matrix(np.identity(nprim))

    # h = vector_constraint_ineq
    vector_constraint_ineq = cvxopt.matrix(0.0, (nprim, 1))

    # Linear equality constraints :
    # Ax = b with x=(c_(a,k))_{k=1..Ka} ; A = (1...1) and b = Na = (Na)
    matrix_constraint_eq = cvxopt.matrix(1.0, (1, nprim))

    r = rgrid.points
    # avoid too large r
    r = np.clip(r, 1e-100, 1e10)
    # N_a : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k) = N_a
    N_a = rgrid.integrate(4 * np.pi * r**2, rho)
    vector_constraint_eq = cvxopt.matrix(N_a, (1, 1))

    # Use optimized x to calculate Gaussian functions
    gauss_funcs = np.array(
        [get_gauss_function(1.0, alphas[k], r) for k in range(nprim)]
    )

    def F(x=None, z=None):
        # x is the optimized coefficients
        if x is None:
            # For the initial step, this should be propars
            return 0, cvxopt.matrix(propars[:])

        x = np.clip(x, 1e-6, None)  # Replace values < 1e-6 with 1e-6

        # Use optimized to calculate density from each Gaussian function.
        gauss_pros = np.array(
            [get_gauss_function(x[k], alphas[k], r) for k in range(nprim)]
        )
        pro = gauss_pros.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)

        f = -rgrid.integrate(4 * np.pi * r**2, rho * np.log(pro))
        df = np.zeros((1, nprim), float)
        for i in range(nprim):
            df[0, i] = -rgrid.integrate(4 * np.pi * r**2 * rho * gauss_funcs[i] / pro)
        df = cvxopt.matrix(df)

        if z is None:
            return f, df

        hess = np.zeros((nprim, nprim), float)
        for i in range(nprim):
            for j in range(i, nprim):
                hess[i, j] = rgrid.integrate(
                    4 * np.pi * r**2,
                    rho * gauss_funcs[i] * gauss_funcs[j] / pro**2,
                )
                hess[j, i] = hess[i, j]
        hess = z[0] * cvxopt.matrix(hess)
        return f, df, hess

    opt_CVX = cvxopt.solvers.cp(
        F,
        G=matrix_constraint_ineq,
        h=vector_constraint_ineq,
        A=matrix_constraint_eq,
        b=vector_constraint_eq,
        verbose=verbose,
        reltol=threshold,
        options={"show_progress": log.do_medium},
    )

    optimized_res = opt_CVX["x"]
    if not (np.asarray(optimized_res) > 0).all() and log.do_warning:
        log("Not all values are positive!")

    if np.sum(optimized_res) - N_a >= 1e-8 and log.do_warning:
        log("The sum of results is not equal to N_a!")

    new_propars = np.asarray(opt_CVX["x"]).flatten()
    return new_propars


def _opt_propars_with_lisa_method_fast(
    rho, propars, rgrid, alphas, threshold, verbose=False
):
    nprim = len(propars)
    matrix_constraint_ineq = -cvxopt.matrix(np.identity(nprim))
    vector_constraint_ineq = cvxopt.matrix(0.0, (nprim, 1))
    matrix_constraint_eq = cvxopt.matrix(1.0, (1, nprim))

    r = rgrid.points
    weights = rgrid.weights
    r = np.clip(r, 1e-100, 1e10)
    N_a = rgrid.integrate(4 * np.pi * r**2, rho)
    vector_constraint_eq = cvxopt.matrix(N_a, (1, 1))

    # Precomputed Gaussian functions
    gauss_funcs = np.array(
        [get_gauss_function(1.0, alphas[k], r) for k in range(nprim)]
    )
    integrand_mult = 4 * np.pi * r**2 * weights

    def F(x=None, z=None):
        if x is None:
            return 0, cvxopt.matrix(propars[:])

        x = np.clip(x, 1e-6, None).flatten()
        gauss_pros = gauss_funcs * x[:, None]
        pro = gauss_pros.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)

        f = -rgrid.integrate(integrand_mult, rho * np.log(pro))

        tmp_grad = integrand_mult * rho / pro
        df = -np.sum(tmp_grad[None, :] * gauss_funcs, axis=1).reshape((1, nprim))
        df = cvxopt.matrix(df)

        if z is None:
            return f, df

        tmp_hess = tmp_grad / pro
        hess = np.sum(
            tmp_hess[None, None, :] * gauss_funcs[:, None, :] * gauss_funcs[None, :, :],
            axis=-1,
        )
        hess = z[0] * cvxopt.matrix(hess)
        return f, df, hess

    opt_CVX = cvxopt.solvers.cp(
        F,
        G=matrix_constraint_ineq,
        h=vector_constraint_ineq,
        A=matrix_constraint_eq,
        b=vector_constraint_eq,
        verbose=verbose,
        reltol=threshold,
        options={"show_progress": log.do_medium},
    )

    optimized_res = opt_CVX["x"]
    if not (np.asarray(optimized_res) > 0).all() and log.do_warning:
        log("Not all values are positive!")

    if np.sum(optimized_res) - N_a >= 1e-8 and log.do_warning:
        log("The sum of results is not equal to N_a!")

    new_propars = np.asarray(opt_CVX["x"]).flatten()
    return new_propars


def _opt_propars_with_lisa_method_scipy(
    rho, propars, rgrid, alphas, threshold, verbose=False
):
    nprim = len(propars)

    r = rgrid.points
    weights = rgrid.weights
    r = np.clip(r, 1e-100, 1e10)
    N_a = rgrid.integrate(4 * np.pi * r**2, rho)
    constraint = LinearConstraint(np.ones((1, nprim)), N_a, N_a)

    # Precomputed Gaussian functions
    gauss_funcs = np.array(
        [get_gauss_function(1.0, alphas[k], r) for k in range(nprim)]
    )
    integrand_mult = 4 * np.pi * r**2 * weights * rho

    def F(x=None):
        gauss_pros = gauss_funcs * x[:, None]
        pro = gauss_pros.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)
        f = -np.sum(integrand_mult * np.log(pro))
        tmp_grad = integrand_mult / pro
        df = -np.sum(tmp_grad[None, :] * gauss_funcs, axis=1)
        return f, df

    bounds = [(1e-6, 200)] * nprim
    opt_res = minimize(
        F,
        x0=propars,
        method="trust-constr",
        jac=True,
        bounds=bounds,
        constraints=constraint,
        hess="3-point",
        options={"gtol": 1e-8, "maxiter": 5000},
    )

    if not opt_res.success:
        raise RuntimeError("Convergence failure.")

    optimized_res = opt_res["x"]
    if not (np.asarray(optimized_res) > 0).all() and log.do_warning:
        log("Not all values are positive!")

    if np.sum(optimized_res) - N_a >= 1e-8 and log.do_warning:
        log("The sum of results is not equal to N_a!")
    return optimized_res


def _opt_propars_with_lisa_method_scipy_no_constr(
    rho, propars, rgrid, alphas, threshold, verbose=False
):
    nprim = len(propars)

    r = rgrid.points
    weights = rgrid.weights
    r = np.clip(r, 1e-100, 1e10)
    N_a = rgrid.integrate(4 * np.pi * r**2, rho)
    # constraint = LinearConstraint(np.ones((1, nprim)), N_a, N_a)

    # Precomputed Gaussian functions
    gauss_funcs = np.array(
        [get_gauss_function(1.0, alphas[k], r) for k in range(nprim)]
    )
    integrand_mult = 4 * np.pi * r**2 * weights * rho

    def F(x=None):
        gauss_pros = gauss_funcs * x[:, None]
        pro = gauss_pros.sum(axis=0)
        pro = np.clip(pro, 1e-100, np.inf)
        f = -np.sum(integrand_mult * np.log(pro)) - (N_a - np.sum(x))
        tmp_grad = integrand_mult / pro
        df = -np.sum(tmp_grad[None, :] * gauss_funcs, axis=1) + 1
        return f, df

    bounds = [(1e-6, 200)] * nprim
    opt_res = minimize(
        F,
        x0=propars,
        method="trust-constr",
        jac=True,
        bounds=bounds,
        constraints=None,
        hess="3-point",
        options={"gtol": 1e-8, "maxiter": 5000},
    )

    if not opt_res.success:
        raise RuntimeError("Convergence failure.")

    optimized_res = opt_res["x"]
    if not (np.asarray(optimized_res) > 0).all() and log.do_warning:
        log("Not all values are positive!")

    if np.sum(optimized_res) - N_a >= 1e-4 and log.do_warning:
        log("The sum of results is not equal to N_a!")
    return optimized_res


def _solver_comparison(rho, propars, rgrid, alphas, threshold):
    propars_lisa = _opt_propars_with_lisa_method_fast(
        rho, propars, rgrid, alphas, threshold
    )
    propars_lisa = np.clip(propars_lisa, 0, np.inf)

    propars_lagrangian = _opt_propars_with_mbis_lagrangian(
        rho, propars, rgrid, alphas, threshold
    )
    propars_lagrangian = np.clip(propars_lagrangian, 0, np.inf)
    print("propars_lisa:")
    print(propars_lisa)
    print(np.sum(propars_lisa))
    print("propars_lagrangian:")
    print(propars_lagrangian)
    print(np.sum(propars_lagrangian))
    print("*" * 80)
    assert np.allclose(propars_lisa, propars_lagrangian, atol=1e-2)
    return propars_lisa
    # return propars_lagrangian
