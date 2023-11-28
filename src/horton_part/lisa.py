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
from scipy.linalg import solve, LinAlgWarning, eigh
from scipy.sparse.linalg import spsolve
from scipy.sparse import SparseEfficiencyWarning
from scipy.optimize import (
    minimize,
    LinearConstraint,
    fsolve,
    SR1,
)
import warnings
import time
from .log import log, biblio
from .gisa import GaussianIterativeStockholderWPart
from .cache import just_once
from .basis import compute_quantities, check_pro_atom_parameters

# Suppress specific warning
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)


__all__ = [
    "LinearIterativeStockholderWPart",
    "opt_propars_fixed_points_sc",
    "opt_propars_fixed_points_diis",
    "opt_propars_fixed_points_fslove",
    "opt_propars_fixed_points_newton",
    "opt_propars_minimization_trust_constr",
    "opt_propars_minimization_fast",
    "opt_propars_minimization_no_constr",
]


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
        local_grid_radius=8.0,
        solver=1,
        diis_size=8,
        basis_func_type="gauss",
        use_global_method=False,
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
        self.use_global_method = use_global_method
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
            local_grid_radius,
            solver,
            basis_func_type,
        )

    def _init_log_scheme(self):
        if log.do_medium:
            info_list = [
                ("Scheme", "Linear Iterative Stockholder"),
                ("Outer loop convergence threshold", "%.1e" % self._threshold),
            ]
            if not self.use_global_method:
                info_list.append(
                    (
                        "Inner loop convergence threshold",
                        "%.1e" % self._inner_threshold,
                    )
                )
                info_list.append(("Using global ISA", False))
            else:
                info_list.append(("Using global ISA", True))

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
            log.deflist(info_list)
            biblio.cite(
                "Benda2022", "the use of Linear Iterative Stockholder partitioning"
            )

    @just_once
    def do_partitioning(self):
        if not self.use_global_method:
            return GaussianIterativeStockholderWPart.do_partitioning(self)
        else:
            new = any(("at_weights", i) not in self.cache for i in range(self.natom))
            new |= "niter" not in self.cache
            if new:
                self._init_propars()
                t0 = time.time()
                if self._solver in [1, 101]:
                    new_propars = self._update_propars_lisa_101_globally()
                elif self._solver == 104:
                    warnings.warn(
                        "The slolver 104 with allowing negative parameters problematic, "
                        "because the negative density could be easily found."
                    )
                    new_propars = self._update_propars_lisa_101_globally(
                        allow_neg_pars=True
                    )
                elif self._solver in [2, 201]:
                    new_propars = self._update_propars_lisa_201_globally()
                elif self._solver in [202, 206]:
                    warnings.warn(
                        "The slolver 206 with allowing negative parameters problematic, "
                        "because the negative density could be easily found."
                    )
                    new_propars = self._update_propars_lisa_206_globally(self.diis_size)
                elif self._solver in [3, 301]:
                    new_propars = self._update_propars_lisa_301_globally(
                        gtol=self._threshold, allow_neg_pars=False
                    )
                elif self._solver == 302:
                    warnings.warn(
                        "The slolver 302 with allowing negative parameters problematic, "
                        "because the negative density could be easily found."
                    )
                    new_propars = self._update_propars_lisa_301_globally(
                        gtol=self._threshold
                    )
                else:
                    raise NotImplementedError

                t1 = time.time()
                print(f"Time usage for partitioning: {t1-t0:.2f} s")
                propars = self.cache.load("propars")
                propars[:] = new_propars

                self.update_at_weights()
                # compute the new charge
                charges = self.cache.load("charges", alloc=self.natom, tags="o")[0]
                for iatom in range(self.natom):
                    at_weights = self.cache.load("at_weights", iatom)
                    dens = self.get_moldens(iatom)
                    atgrid = self.get_grid(iatom)
                    spline = atgrid.spherical_average(at_weights * dens)
                    r = atgrid.rgrid.points
                    spherical_average = spline(r)
                    # r = np.clip(atgrid.rgrid.points, 1e-20, 1e10)
                    # spherical_average = np.clip(spline(r), 1e-20, np.inf)
                    pseudo_population = atgrid.rgrid.integrate(
                        4 * np.pi * r**2 * spherical_average
                    )
                    charges[iatom] = self.pseudo_numbers[iatom] - pseudo_population

    def compute_promol_dens(self, propars):
        self.eval_pro_shells_lisa_101()
        rho0 = np.einsum("np,n->p", self.cache.load("pro_shells"), propars)
        # rho0 = np.clip(rho0, 1e-20, None)
        return rho0

    @just_once
    def eval_pro_shells_lisa_101(self):
        self.compute_local_grids()
        nshell = len(self.cache.load("propars"))
        pro_shells = self.cache.load("pro_shells", alloc=(nshell, self.grid.size))[0]
        pro_shells[:] = 0.0
        centers = self.cache.load("pro_shell_centers", alloc=(nshell, 3))[0]

        ishell = 0
        for a in range(self.natom):
            exponents = self.bs_helper.load_exponent(self.numbers[a])
            centers[a] = self.coordinates[a, :]
            for exp in exponents:
                g_ai = self.bs_helper.compute_proshell_dens(
                    1.0, exp, self.radial_dists[a], 0
                )
                # g_ai = np.clip(g_ai, 1e-100, np.inf)
                pro_shells[ishell, self.local_grids[a].indices] = g_ai

                ishell += 1

        rho = self._moldens
        rho_x_pro_shells = self.cache.load(
            "rho*pro_shells", alloc=(nshell, self.grid.size)
        )[0]
        rho_x_pro_shells[:] = rho[None, :] * pro_shells[:, :]

    def _update_propars_lisa_101_globally(self, allow_neg_pars=False):
        rho = self._moldens
        propars = self.cache.load("propars")

        nb_par = len(propars)
        if allow_neg_pars:
            matrix_constraint_ineq = None
            vector_constraint_ineq = None
        else:
            matrix_constraint_ineq = -cvxopt.matrix(np.identity(nb_par))
            vector_constraint_ineq = cvxopt.matrix(0.0, (nb_par, 1))

        matrix_constraint_eq = cvxopt.matrix(1.0, (1, nb_par))
        mol_pop = self.grid.integrate(rho)
        vector_constraint_eq = cvxopt.matrix(mol_pop, (1, 1))

        def obj_func(x=None, z=None):
            if x is None:
                return 0, cvxopt.matrix(propars[:])
            x = np.asarray(x).flatten()
            rho0 = self.compute_promol_dens(x)
            check_pro_atom_parameters(x, pro_atom_density=rho0)

            if z is None:
                f, df = self._working_matrix(rho, rho0, nb_par, 1)
                df = df.reshape((1, nb_par))
                return f, cvxopt.matrix(df)

            f, df, hess = self._working_matrix(rho, rho0, nb_par, 2)
            df = df.reshape((1, nb_par))
            hess = z[0] * cvxopt.matrix(hess)
            return f, cvxopt.matrix(df), hess

        opt_CVX = cvxopt.solvers.cp(
            obj_func,
            G=matrix_constraint_ineq,
            h=vector_constraint_ineq,
            A=matrix_constraint_eq,
            b=vector_constraint_eq,
            verbose=True,
            options={"show_progress": log.do_medium, "feastol": self._threshold},
        )

        optimized_res = opt_CVX["x"]
        propars[:] = np.asarray(optimized_res).flatten()
        if (propars < 0).any():
            warnings.warn("Not all values are positive!")
        if np.sum(propars) - mol_pop >= 1e-8:
            warnings.warn("The sum of results is not equal to N_a!")
        return propars

    def _working_matrix(self, rho, rho0, nb_par, nderiv=0, density_cutoff=1e-15):
        """
        Compute the working matrix for optimization, based on molecular and pro-molecule densities, and pro-atom parameters.

        Parameters
        ----------
        rho : ndarray
            Molecular density on grids.
        rho0 : ndarray
            Pro-molecule density on grids.
        nb_par: int
            The number of parameters in total.
        nderiv : int, optional
            Order of derivatives to be computed (0, 1, or 2). Default is 0.

        Returns
        -------
        float or tuple
            If nderiv is 0, returns the objective function value.
            If nderiv is 1, returns a tuple (objective function value, gradient).
            If nderiv is 2, returns a tuple (objective function value, gradient, hessian).

        Raises
        ------
        ValueError
            If `nderiv` is not 0, 1, or 2.

        Notes
        -----
        This function computes the objective function and its derivatives for a given set of molecular
        densities and pro-atom parameters. The function supports up to the second derivative calculation.
        Numerical stability for division by `rho0` is ensured by adding a small constant to `rho0`.

        """
        sick = (rho < density_cutoff) | (rho0 < density_cutoff)
        with np.errstate(all="ignore"):
            lnpro = np.log(rho0)
        lnpro[sick] = 0.0

        objective_function = -self.grid.integrate(rho * lnpro)

        if nderiv == 0:
            return objective_function

        pro_shells = self.cache.load("pro_shells")
        rho_x_pro_shells = self.cache.load("rho*pro_shells")
        centers = self.cache.load("pro_shell_centers")

        gradient = np.zeros((nb_par,))
        hessian = np.zeros((nb_par, nb_par))

        for i in range(nb_par):
            with np.errstate(all="ignore"):
                df_integrand = rho_x_pro_shells[i, :] / rho0
            df_integrand[sick] = 0.0
            gradient[i] = -self.grid.integrate(df_integrand)

            if nderiv > 1:
                for j in range(i, nb_par):
                    if np.linalg.norm(centers[i] - centers[j]) > self.local_grid_radius:
                        hessian[i, j] = 0
                    else:
                        with np.errstate(all="ignore"):
                            hess_integrand = df_integrand * pro_shells[j, :] / rho0
                        hess_integrand[sick] = 0.0
                        hessian[i, j] = self.grid.integrate(hess_integrand)

                    hessian[j, i] = hessian[i, j]

        if nderiv == 1:
            return objective_function, gradient
        elif nderiv == 2:
            return objective_function, gradient, hessian
        else:
            raise ValueError(
                f"nderiv value of {nderiv} is not supported. Only 0, 1, or 2 are valid."
            )

    def _update_propars_lisa_301_globally(
        self, gtol, maxiter=1000, allow_neg_pars=True
    ):
        """Optimize the promodel using the trust-constr minimizer from SciPy."""
        rho = self._moldens

        # Compute the total population
        pop = self.grid.integrate(rho)
        print("Integral of density:", pop)
        pars0 = self.cache.load("propars")
        nb_par = len(pars0)

        def cost_grad(x):
            rho0 = self.compute_promol_dens(x)
            f, df = self._working_matrix(rho, rho0, nb_par, 1)
            # f += self.grid.integrate(rho * np.log(rho))
            f += self.grid.integrate(rho0) - pop
            df += 1
            return f, df

        # Optimize parameters within the bounds.
        if allow_neg_pars:
            bounds = None
            constraint = LinearConstraint(np.ones((1, nb_par)), pop, pop)
            hess = SR1()
        else:
            bounds = [(0.0, 200)] * len(pars0)
            constraint = None
            hess = SR1()

        rho0_history = []

        def callback(_current_pars, opt_result):
            rho0 = self.compute_promol_dens(_current_pars)
            rho0_history.append(rho0)

            if len(rho0_history) >= 2:
                pre_rho0 = rho0_history[-2]
                change = np.sqrt(self.grid.integrate((rho0 - pre_rho0) ** 2))
                return change < self._threshold and opt_result["status"]

        optresult = minimize(
            cost_grad,
            pars0,
            method="trust-constr",
            jac=True,
            hess=hess,
            bounds=bounds,
            constraints=constraint,
            callback=callback,
            options={"gtol": gtol, "maxiter": maxiter, "verbose": 5},
        )

        # Check for convergence.
        print(f'Optimizer message: "{optresult.message}"')
        if not optresult.success:
            raise RuntimeError("Convergence failure.")

        rho0 = self.compute_promol_dens(optresult.x)
        constrain = self.grid.integrate(rho0) - pop
        print(f"Constraint: {constrain}")
        print("Optimized parameters: ")
        print(optresult.x)
        return optresult.x

    @just_once
    def eval_pro_shells_lisa_201(self):
        self.compute_local_grids()
        for a in range(self.natom):
            exponents = self.bs_helper.load_exponent(self.numbers[a])
            indices = self.local_grids[a].indices
            for i, exp in enumerate(exponents):
                g_ai = self.cache.load("pro-shell", a, i, alloc=len(indices))[0]
                g_ai[:] = self.bs_helper.compute_proshell_dens(
                    1.0, exp, self.radial_dists[a], 0
                )
                # g_ai[:] = np.clip(tmp, 1e-20, np.inf)

    def _update_propars_lisa_201_globally(self, density_cutoff=1e-15):
        # 1. load molecular and pro-molecule density from cache
        rho = self._moldens
        self.eval_pro_shells_lisa_201()
        all_propars = self.cache.load("propars")
        old_propars = all_propars.copy()

        print("Iteration       Change")

        counter = 0
        while True:
            old_rho0 = self.compute_promol_dens(old_propars)
            sick = (rho < density_cutoff) | (old_rho0 < density_cutoff)

            ishell = 0
            for iatom in range(self.natom):
                # 2. load old propars
                propars = all_propars[self._ranges[iatom] : self._ranges[iatom + 1]]
                alphas = self.bs_helper.load_exponent(self.numbers[iatom])

                # 3. compute basis functions on molecule grid
                new_propars = []
                local_grid = self.local_grids[iatom]
                indices = local_grid.indices
                for k, (pop, alpha) in enumerate(zip(propars.copy(), alphas)):
                    g_ak = self.cache.load(("pro-shell", iatom, k))
                    rho0_ak = g_ak * pop

                    with np.errstate(all="ignore"):
                        integrand = rho[indices] * rho0_ak / old_rho0[indices]
                    integrand[sick[indices]] = 0.0

                    new_propars.append(local_grid.integrate(integrand))
                    ishell += 1

                # 4. get new propars using fixed-points
                propars[:] = np.asarray(new_propars)

            rho0 = self.compute_promol_dens(all_propars)
            change = np.sqrt(self.grid.integrate((rho0 - old_rho0) ** 2))
            if counter % 10 == 0:
                print("%9i   %10.5e" % (counter, change))
            if change < self._threshold:
                print("%9i   %10.5e" % (counter, change))
                break
            old_propars = all_propars.copy()
            counter += 1
        return all_propars

    def _update_propars_lisa_206_globally(self, diis_size=8, density_cutoff=1e-15):
        # 1. load molecular and pro-molecule density from cache
        rho = self._moldens
        self.eval_pro_shells_lisa_201()
        all_propars = self.cache.load("propars")

        print("Iteration       Change")

        history_diis = []
        history_propars = []
        start_diis_iter = diis_size + 1

        for counter in range(1000):
            old_rho0 = self.compute_promol_dens(all_propars)
            sick = (rho < density_cutoff) | (old_rho0 < density_cutoff)

            all_fun_vals = np.zeros_like(all_propars)
            ishell = 0
            for iatom in range(self.natom):
                # 2. load old propars
                propars = all_propars[self._ranges[iatom] : self._ranges[iatom + 1]]
                alphas = self.bs_helper.load_exponent(self.numbers[iatom])

                # 3. compute basis functions on molecule grid
                fun_val = []
                local_grid = self.local_grids[iatom]
                indices = local_grid.indices
                for k, (pop, alpha) in enumerate(zip(propars.copy(), alphas)):
                    g_ak = self.cache.load(("pro-shell", iatom, k))
                    rho0_ak = g_ak * pop

                    with np.errstate(all="ignore"):
                        integrand = rho[indices] * rho0_ak / old_rho0[indices]
                    integrand[sick[indices]] = 0.0

                    fun_val.append(local_grid.integrate(integrand))
                    ishell += 1

                # 4. get new propars using fixed-points
                # new_propars[:] = np.asarray(fun_val)
                all_fun_vals[
                    self._ranges[iatom] : self._ranges[iatom + 1]
                ] = np.asarray(fun_val)

            history_propars.append(all_propars)

            # Build DIIS Residual
            diis_r = all_propars - all_fun_vals
            rho0 = self.compute_promol_dens(all_fun_vals)
            change = np.sqrt(self.grid.integrate((rho0 - old_rho0) ** 2))

            # Compute drms
            drms = np.linalg.norm(diis_r)

            if log.do_medium:
                log(f"           {counter:<4}    {drms:.6E}")

            if change < self._threshold:
                return all_propars

            # if drms < self._threshold:
            #     return all_propars

            # Append trail & residual vectors to lists
            if counter >= start_diis_iter - diis_size:
                history_propars.append(all_fun_vals)
                history_diis.append(diis_r)

            if counter >= start_diis_iter:
                # TODO: this doesn't work for global DIIS
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
                all_propars = np.einsum(
                    "i, ip->p", coeff[:-1], np.asarray(propars_prev)
                )

                if (np.isnan(all_propars)).any():
                    all_propars = all_fun_vals
            else:
                all_propars = all_fun_vals

            if not np.all(all_propars >= -1e-10).all():
                log("negative c_ak found!")

        raise RuntimeError("Error: inner iteration is not converge!")

    def _finalize_propars(self):
        if not self.use_global_method:
            return GaussianIterativeStockholderWPart._finalize_propars(self)
        else:
            self._cache.load("charges")
            pass

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
                bs_funcs,
                rho,
                propars,
                points,
                weights,
                threshold,
                density_cutoff=density_cutoff,
            )
        if self._solver == 104:
            # no robust: HF
            return opt_propars_minimization_fast(
                bs_funcs,
                rho,
                propars,
                points,
                weights,
                threshold,
                density_cutoff=density_cutoff,
                allow_neg_params=True,
            )
        elif self._solver in [2, 201]:
            return opt_propars_fixed_points_sc(
                bs_funcs,
                rho,
                propars,
                points,
                weights,
                threshold,
                density_cutoff=density_cutoff,
            )
        elif self._solver in [202, 206]:
            return opt_propars_fixed_points_diis(
                bs_funcs,
                rho,
                propars,
                points,
                weights,
                threshold,
                density_cutoff=density_cutoff,
                diis_size=self.diis_size,
            )
        elif self._solver == 203:
            # not robust
            return opt_propars_fixed_points_newton(
                bs_funcs,
                rho,
                propars,
                points,
                weights,
                threshold,
                density_cutoff=density_cutoff,
            )
        elif self._solver == 204:
            return opt_propars_fixed_points_fslove(
                bs_funcs,
                rho,
                propars,
                points,
                weights,
                threshold,
                density_cutoff=density_cutoff,
            )
        elif self._solver in [3, 301]:
            # same as LISA-102 but with constraint implicitly
            return opt_propars_minimization_no_constr(
                bs_funcs,
                rho,
                propars,
                points,
                weights,
                threshold,
                density_cutoff=density_cutoff,
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


def opt_propars_fixed_points_sc(
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
    oldpro = None
    if log.do_medium:
        log("            Iter.    Change    ")
        log("            -----    ------    ")
    for irep in range(int(1e10)):
        pro_shells, pro, sick, lnpro, ratio, lnratio = compute_quantities(
            rho, propars, bs_funcs, density_cutoff
        )

        # the partitions and the updated parameters
        propars[:] = np.einsum(
            "p,ip->i", 4 * np.pi * points**2 * weights, pro_shells * ratio
        )
        check_pro_atom_parameters(propars)

        # check for convergence
        if oldpro is None:
            change = 1e100
        else:
            error = oldpro - pro
            change = np.sqrt(
                np.einsum("i,i,i->", 4 * np.pi * points**2 * weights, error, error)
            )
        if log.do_medium:
            log(f"            {irep+1:<4}    {change:.3e}")
        if change < threshold:
            return propars
        oldpro = pro
    raise RuntimeError("Error: Inner iteration is not converge!")


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
    if nb_coeff > 2 and np.allclose(
        coeffs[:-1], np.ones_like(coeffs[:-1]) / nb_coeff, rtol=1e-3
    ):
        turn_off_diis = True
        print("turn off DIIS")

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

    if log.do_medium:
        log("            Iter.    dRMS      ")
        log("            -----    ------    ")

    for irep in range(1000):
        pro_shells, pro, sick, lnpro, ratio, lnratio = compute_quantities(
            rho, propars, bs_funcs, density_cutoff
        )

        integrands = pro_shells * ratio
        fun_val = np.einsum("ip,p->i", integrands, 4 * np.pi * points**2 * weights)

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
    int_weights = 4 * np.pi * points**2 * weights

    if log.do_medium:
        log("            Iter.    Change    ")
        log("            -----    ------    ")

    oldpro = None
    change = 1e100
    for irep in range(1000):
        pro_shells, pro, sick, lnpro, ratio, lnratio = compute_quantities(
            rho, propars, bs_funcs, density_cutoff
        )
        integrand = bs_funcs * ratio
        check_pro_atom_parameters(
            propars, total_population=float(np.sum(pro)), pro_atom_density=pro
        )

        # check for convergence
        if oldpro is not None:
            error = oldpro - pro
            change = np.sqrt(np.einsum("i,i,i->", int_weights, error, error))
        if log.do_medium:
            log(f"            {irep+1:<4}    {change:.3e}")
        if change < threshold:
            return propars

        # update propars
        with np.errstate(all="ignore"):
            grad_integrand = integrand / pro
        grad_integrand[:, sick] = 0.0

        grad = np.einsum("kp, jp, p->kj", grad_integrand, bs_funcs, int_weights)
        h = 1 - np.einsum("kp,p->k", integrand, int_weights)
        delta = solve(grad, -h, assume_a="sym")
        propars += delta
        oldpro = pro
    raise RuntimeError("Inner loop: Newton does not converge!")


def opt_propars_fixed_points_fslove(
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
    threshold:
        Threshold for convergence.

    Returns
    -------

    """
    int_weights = 4 * np.pi * points**2 * weights

    def obj_func(vars):
        pro_shells, pro, sick, lnpro, ratio, lnratio = compute_quantities(
            rho, vars, bs_funcs, density_cutoff
        )
        return 1 - np.einsum("kp,p->k", bs_funcs * ratio, int_weights)

    def fprime(vars):
        pro_shells, pro, sick, lnpro, ratio, lnratio = compute_quantities(
            rho, vars, bs_funcs, density_cutoff
        )
        with np.errstate(all="ignore"):
            integrand = bs_funcs * ratio / pro
        integrand[:, sick] = 0
        grad = np.einsum("kp, jp, p->kj", integrand, bs_funcs, int_weights)
        return grad

    # TODO: xtol is relative error not absolute
    new_propars, infodict, iter, msg = fsolve(
        obj_func, propars, fprime=fprime, xtol=threshold, maxfev=1000, full_output=True
    )
    if log.do_medium:
        print(f"iter: {iter}")
        print(msg)

    # without fprime, it will get in trouble for some atoms, e.g., HF.
    # solution = fsolve(func, propars, xtol=threshold)

    check_pro_atom_parameters(
        new_propars, bs_funcs, total_population=int(np.sum(int_weights, rho))
    )
    return new_propars


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

    integrand_mult = 4 * np.pi * points**2 * weights * rho
    pop = np.sum(integrand_mult)
    matrix_constraint_eq = cvxopt.matrix(1.0, (1, nprim))
    vector_constraint_eq = cvxopt.matrix(pop, (1, 1))

    def obj_func(x=None, z=None):
        if x is None:
            return 0, cvxopt.matrix(propars[:])

        pro_shells, pro, sick, lnpro, ratio, lnratio = compute_quantities(
            rho, x, bs_funcs, density_cutoff
        )

        f = -np.sum(integrand_mult * lnpro)

        with np.errstate(all="ignore"):
            tmp_grad = integrand_mult / pro
        tmp_grad[sick] = 0.0

        df = -np.sum(tmp_grad[None, :] * bs_funcs, axis=1).reshape((1, nprim))
        df = cvxopt.matrix(df)

        if z is None:
            return f, df

        with np.errstate(all="ignore"):
            tmp_hess = tmp_grad / pro
        tmp_hess[sick] = 0.0

        hess = np.sum(
            tmp_hess[None, None, :] * bs_funcs[:, None, :] * bs_funcs[None, :, :],
            axis=-1,
        )
        hess = z[0] * cvxopt.matrix(hess)
        return f, df, hess

    opt_CVX = cvxopt.solvers.cp(
        obj_func,
        G=matrix_constraint_ineq,
        h=vector_constraint_ineq,
        A=matrix_constraint_eq,
        b=vector_constraint_eq,
        options={"show_progress": log.do_medium, "feastol": threshold},
    )

    optimized_res = opt_CVX["x"]
    new_propars = np.asarray(optimized_res).flatten()
    check_pro_atom_parameters(new_propars, bs_funcs)
    return new_propars


def opt_propars_minimization_trust_constr(
    bs_funcs, rho, propars, points, weights, threshold, density_cutoff
):
    nprim = len(propars)
    integrand_mult = 4 * np.pi * points**2 * weights * rho
    pop = np.sum(integrand_mult)
    constraint = LinearConstraint(np.ones((1, nprim)), pop, pop)

    def obj_funcs(x=None):
        pro_shells, pro, sick, lnpro, ratio, lnratio = compute_quantities(
            rho, x, bs_funcs, density_cutoff
        )
        f = -np.sum(integrand_mult * lnpro)
        with np.errstate(all="ignore"):
            grad_integrand = integrand_mult / pro
        grad_integrand[sick] = 0
        df = -np.sum(grad_integrand[None, :] * bs_funcs, axis=1)
        return f, df

    bounds = [(0.0, 200)] * nprim
    opt_res = minimize(
        obj_funcs,
        x0=propars,
        method="trust-constr",
        jac=True,
        bounds=bounds,
        constraints=constraint,
        hess="3-point",
        options={"gtol": threshold, "maxiter": 5000},
    )

    if not opt_res.success:
        raise RuntimeError("Convergence failure.")

    optimized_res = opt_res["x"]
    new_propars = np.asarray(optimized_res).flatten()
    check_pro_atom_parameters(new_propars, bs_funcs, total_population=float(pop))
    return new_propars


def opt_propars_minimization_no_constr(
    bs_funcs, rho, propars, points, weights, threshold, density_cutoff
):
    nprim = len(propars)
    integrand_mult = 4 * np.pi * points**2 * weights * rho
    pop = np.sum(integrand_mult)

    def obj_func(x=None):
        pro_shells, pro, sick, lnpro, ratio, lnratio = compute_quantities(
            rho, x, bs_funcs, density_cutoff
        )
        f = -np.sum(integrand_mult * lnpro) - (pop - np.sum(x))
        with np.errstate(all="ignore"):
            grad_integrand = integrand_mult / pro
        grad_integrand[sick] = 0
        df = -np.sum(grad_integrand[None, :] * bs_funcs, axis=1) + 1
        return f, df

    bounds = [(0.0, 200)] * nprim
    opt_res = minimize(
        obj_func,
        x0=propars,
        method="trust-constr",
        jac=True,
        bounds=bounds,
        constraints=None,
        hess="3-point",
        options={"gtol": threshold, "maxiter": 5000},
    )

    if not opt_res.success:
        raise RuntimeError("Convergence failure.")

    new_propars = np.asarray(opt_res["x"]).flatten()
    check_pro_atom_parameters(new_propars, bs_funcs, total_population=float(pop))
    return new_propars
