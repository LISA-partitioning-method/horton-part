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
"""
Module for Global Linear Iterative Stockholder Analysis (LISA-G) partitioning scheme.
"""

import numpy as np
import cvxopt
from scipy.linalg import LinAlgWarning
from scipy.sparse.linalg import spsolve
from scipy.sparse import SparseEfficiencyWarning
from scipy.optimize import minimize, LinearConstraint, SR1
import warnings
import time
import logging

from .core.cache import just_once
from .utils import (
    check_pro_atom_parameters,
)
from .lisa import LinearISAWPart
from .core.logging import deflist

# Suppress specific warning
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

__all__ = [
    "AbstractGlobalLinearISAWPart",
    "GlobalLinearISA101WPart",
    "GlobalLinearISA104WPart",
    "GlobalLinearISA201WPart",
    "GlobalLinearISA206WPart",
    "GlobalLinearISA301WPart",
    "GlobalLinearISA302WPart",
]

logger = logging.getLogger(__name__)


class AbstractGlobalLinearISAWPart(LinearISAWPart):
    name = "lisa_g"
    use_global_method = True
    density_cutoff = 1e-15
    allow_neg_pars = False

    def _init_log_scheme(self):
        # if log.do_medium:
        info_list = [
            ("Scheme", "Linear Iterative Stockholder"),
            ("Outer loop convergence threshold", "%.1e" % self._threshold),
            ("Using global ISA", self.use_global_method),
        ]

        info_list.extend(
            [
                ("Maximum outer iterations", self._maxiter),
                ("lmax", self._lmax),
                ("Solver", self._solver),
                ("Basis function type", self.func_type),
                ("Local grid radius", self._local_grid_radius),
                ("Allow negative parameters", self.allow_neg_pars),
            ]
        )
        if self._solver in [202, 206]:
            info_list.append(("DIIS size", self.diis_size))
        deflist(logger, info_list)
        # biblio.cite(
        #     "Benda2022", "the use of Linear Iterative Stockholder partitioning"
        # )

    def _finalize_propars(self):
        self._cache.load("charges")

    # def _opt_propars(self):
    #     if self._solver in [1, 101]:
    #         new_propars = self._update_propars_lisa_101()
    #     elif self._solver == 104:
    #         warnings.warn(
    #             "The slolver 104 with allowing negative parameters problematic, "
    #             "because the negative density could be easily found."
    #         )
    #         new_propars = self._update_propars_lisa_101(allow_neg_pars=True)
    #     elif self._solver in [2, 201]:
    #         new_propars = self._update_propars_lisa_201()
    #     elif self._solver in [202, 206]:
    #         warnings.warn(
    #             "The slolver 206 with allowing negative parameters problematic, "
    #             "because the negative density could be easily found."
    #         )
    #         new_propars = self._update_propars_lisa_206(self.diis_size)
    #     elif self._solver in [3, 301]:
    #         new_propars = self._update_propars_lisa_301(
    #             gtol=self._threshold, allow_neg_pars=False
    #         )
    #     elif self._solver == 302:
    #         warnings.warn(
    #             "The slolver 302 with allowing negative parameters problematic, "
    #             "because the negative density could be easily found."
    #         )
    #         new_propars = self._update_propars_lisa_301(gtol=self._threshold)
    #     else:
    #         raise NotImplementedError
    #     return new_propars

    @just_once
    def do_partitioning(self):
        """Do global partitioning scheme."""
        # Initialize local grids to save resources
        self.initial_local_grids()

        new = any(("at_weights", i) not in self.cache for i in range(self.natom))
        new |= "niter" not in self.cache
        if new:
            # Initialize pro-atom parameters.
            self._init_propars()
            t0 = time.time()
            new_propars = self._opt_propars()
            t1 = time.time()
            logger.info(f"Time usage for partitioning: {t1-t0:.2f} s")
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
                pseudo_population = atgrid.rgrid.integrate(
                    4 * np.pi * r**2 * spherical_average
                )
                charges[iatom] = self.pseudo_numbers[iatom] - pseudo_population

    @just_once
    def eval_pro_shells(self):
        raise NotImplementedError

    def _opt_propars(self):
        raise NotImplementedError


class GlobalLinearISA101WPart(AbstractGlobalLinearISAWPart):
    name = "lisa_g_101"
    allow_neg_pars = False

    def compute_promol_dens(self, propars):
        """Compute pro-molecule density based on pro-atom parameters."""
        self.eval_pro_shells()
        return np.einsum("np,n->p", self.cache.load("pro_shells"), propars)

    @just_once
    def eval_pro_shells(self):
        """Evaluate pro-shell functions on molecular grids."""
        nshell = len(self.cache.load("propars"))
        pro_shells = self.cache.load("pro_shells", alloc=(nshell, self.grid.size))[0]
        pro_shells[:] = 0.0
        centers = self.cache.load("pro_shell_centers", alloc=(nshell, 3))[0]

        index = 0
        for iatom in range(self.natom):
            centers[iatom] = self.coordinates[iatom, :]
            number = self.numbers[iatom]
            for ishell in range(self.bs_helper.get_nshell(number)):
                g_ai = self.cache.load(
                    "pro-shell", iatom, ishell, alloc=len(self.radial_distances[iatom])
                )[0]
                g_ai[:] = self.bs_helper.compute_proshell_dens(
                    number, ishell, 1.0, self.radial_distances[iatom], 0
                )
                pro_shells[index, self.local_grids[iatom].indices] = g_ai
                # g_ai = self.cache.load("pro-shell", iatom, ishell, alloc=len(indices))[
                #     0
                # ]
                # g_ai[:] = self.bs_helper.compute_proshell_dens(
                #     number, ishell, 1.0, self.radial_distances[iatom], 0
                # )

                index += 1

        rho = self._moldens
        rho_x_pro_shells = self.cache.load(
            "rho*pro_shells", alloc=(nshell, self.grid.size)
        )[0]
        rho_x_pro_shells[:] = rho[None, :] * pro_shells[:, :]

    def _opt_propars(self):
        rho = self._moldens
        propars = self.cache.load("propars")

        nb_par = len(propars)
        if self.allow_neg_pars:
            matrix_constraint_ineq = None
            vector_constraint_ineq = None
        else:
            matrix_constraint_ineq = -cvxopt.matrix(np.identity(nb_par))
            vector_constraint_ineq = cvxopt.matrix(0.0, (nb_par, 1))

        matrix_constraint_eq = cvxopt.matrix(1.0, (1, nb_par))
        mol_pop = self.grid.integrate(rho)
        vector_constraint_eq = cvxopt.matrix(mol_pop, (1, 1))

        def obj_func(x=None, z=None):
            """The objective function of the global optimization method."""
            if x is None:
                return 0, cvxopt.matrix(propars[:])
            x = np.asarray(x).flatten()
            rho0 = self.compute_promol_dens(x)
            #
            # Note: the propars and pro-mol density could be negative during the optimization.
            #
            # check_pro_atom_parameters(
            #     x, pro_atom_density=rho0, check_monotonicity=False
            # )

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
            # options={"show_progress": log.do_medium, "feastol": self._threshold},
            options={"show_progress": 3, "feastol": self._threshold},
        )

        propars[:] = np.asarray(opt_CVX["x"]).flatten()
        check_pro_atom_parameters(
            propars,
            pro_atom_density=self.compute_promol_dens(propars),
            total_population=mol_pop,
            check_monotonicity=False,
        )
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
        ratio = np.divide(rho, rho0, out=np.zeros_like(rho), where=~sick)
        ln_ratio = np.log(ratio, out=np.zeros_like(ratio), where=~sick)

        objective_function = self.grid.integrate(rho * ln_ratio)

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


class GlobalLinearISA104WPart(GlobalLinearISA101WPart):
    name = "lisa_g_104"
    allow_neg_pars = True


class GlobalLinearISA301WPart(GlobalLinearISA101WPart):
    name = "lisa_g_301"
    allow_neg_pars = False

    def _opt_propars(self):
        """Optimize the promodel using the trust-constr minimizer from SciPy."""
        rho = self._moldens

        # Compute the total population
        pop = self.grid.integrate(rho)
        logger.info("Integral of density:", pop)
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
        if self.allow_neg_pars:
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
            options={"gtol": self._threshold, "maxiter": self._maxiter, "verbose": 5},
        )

        # Check for convergence.
        logger.info(f'Optimizer message: "{optresult.message}"')
        if not optresult.success:
            raise RuntimeError("Convergence failure.")

        rho0 = self.compute_promol_dens(optresult.x)
        constrain = self.grid.integrate(rho0) - pop
        logger.info(f"Constraint: {constrain}")
        logger.info("Optimized parameters: ")
        logger.info(optresult.x)
        return optresult.x


class GlobalLinearISA302WPart(GlobalLinearISA301WPart):
    name = "lisa_g_302"
    allow_neg_pars = True


class GlobalLinearISA201WPart(GlobalLinearISA101WPart):
    name = "lisa_g_201"

    # @just_once
    # def eval_pro_shells(self):
    #     """Evaluate pro-shell functions on (local) molecular grids for the self-consistent method."""
    #     for iatom in range(self.natom):
    #         number = self.numbers[iatom]
    #         indices = self.local_grids[iatom].indices
    #         for ishell in range(self.bs_helper.get_nshell(number)):
    #             g_ai = self.cache.load("pro-shell", iatom, ishell, alloc=len(indices))[
    #                 0
    #             ]
    #             g_ai[:] = self.bs_helper.compute_proshell_dens(
    #                 number, ishell, 1.0, self.radial_distances[iatom], 0
    #             )

    def _opt_propars(self):
        # 1. load molecular and pro-molecule density from cache
        rho = self._moldens
        self.eval_pro_shells()
        all_propars = self.cache.load("propars")
        old_propars = all_propars.copy()

        logger.info("Iteration       Change")

        counter = 0
        while True:
            old_rho0 = self.compute_promol_dens(old_propars)
            sick = (rho < self.density_cutoff) | (old_rho0 < self.density_cutoff)
            # sick = (rho < density_cutoff * self.natom) | (
            #         old_rho0 < density_cutoff * self.natom
            # )

            ishell = 0
            for iatom in range(self.natom):
                # 2. load old propars
                propars = all_propars[self._ranges[iatom] : self._ranges[iatom + 1]]
                alphas = self.bs_helper.exponents[self.numbers[iatom]]

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

            # rho0 = self.compute_promol_dens(all_propars)
            # change = np.sqrt(self.grid.integrate((rho0 - old_rho0) ** 2))
            change = self.compute_change(all_propars, old_propars)
            if counter % 10 == 0:
                logger.info("%9i   %10.5e" % (counter, change))
            if change < self._threshold:
                logger.info("%9i   %10.5e" % (counter, change))
                break
            old_propars = all_propars.copy()
            counter += 1
        return all_propars


class GlobalLinearISA206WPart(GlobalLinearISA201WPart):
    name = "lisa_g_206"

    def _opt_propars(self):
        # 1. load molecular and pro-molecule density from cache
        rho = self._moldens
        self.eval_pro_shells()
        all_propars = self.cache.load("propars")

        logger.info("Iteration       Change")

        history_diis = []
        history_propars = []
        start_diis_iter = self.diis_size + 1

        for counter in range(1000):
            old_rho0 = self.compute_promol_dens(all_propars)
            sick = (rho < self.density_cutoff) | (old_rho0 < self.density_cutoff)

            all_fun_vals = np.zeros_like(all_propars)
            ishell = 0
            for iatom in range(self.natom):
                # 2. load old propars
                propars = all_propars[self._ranges[iatom] : self._ranges[iatom + 1]]
                alphas = self.bs_helper.exponents[self.numbers[iatom]]

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

            logger.info(f"           {counter:<4}    {drms:.6E}")

            if change < self._threshold:
                return all_propars

            # if drms < self._threshold:
            #     return all_propars

            # Append trail & residual vectors to lists
            if counter >= start_diis_iter - self.diis_size:
                history_propars.append(all_fun_vals)
                history_diis.append(diis_r)

            if counter >= start_diis_iter:
                # TODO: this doesn't work for global DIIS
                # Build B matrix
                propars_prev = history_propars[-self.diis_size :]
                diis_prev = history_diis[-self.diis_size :]

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
                warnings.warn("Negative parameters found!")

        raise RuntimeError("Error: inner iteration is not converge!")
