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
Module for Global Linear Iterative Stockholder Analysis (GL-ISA) partitioning scheme.

**Optimization Problem Schemes**

- Convex optimization (`name` = "glisa_cvxopt")
- Trust-region methods with constraints
    - Implicit constraints (`name` = "glisa_trust_constr")
- Fixed-point methods
    - Alternating method (`name` = "glisa_sc")
    - DIIS (`name` = "glisa_diis")
"""

import time

import cvxopt
import numpy as np
from scipy.optimize import SR1, LinearConstraint, minimize

from horton_part import gisa
from horton_part.core import iterstock

from .algo import cdiis, diis
from .core.cache import just_once
from .core.iterstock import compute_change
from .core.logging import deflist
from .core.stockholder import AbstractStockholderWPart
from .gisa import get_proatom_rho
from .lisa import setup_bs_helper
from .utils import check_pro_atom_parameters

__all__ = ["GlobalLinearISAWPart"]


class GlobalLinearISAWPart(AbstractStockholderWPart):
    name = "glisa"

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
        radius_cutoff=np.inf,
        solver="cvxopt",
        solver_options=None,
        basis_func="gauss",
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
        self._radius_cutoff = radius_cutoff
        self._maxiter = maxiter
        self._threshold = threshold
        self.basis_func = basis_func
        if self.basis_func in ["gauss", "slater"]:
            self._func_type = self.basis_func.upper()
        else:
            self._func_type = "Customized"
        self._bs_helper = None
        self._solver = solver
        self._solver_options = solver_options or {}

        AbstractStockholderWPart.__init__(
            self,
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            True,
            lmax,
            logger,
        )

    @property
    def bs_helper(self):
        """A basis function helper."""
        return setup_bs_helper(self)

    def get_rgrid(self, index):
        """Load radial grid."""
        return self.get_grid(index).rgrid

    def compute_change(self, propars1, propars2):
        """Compute the difference between an old and a new proatoms"""
        return compute_change(self, propars1, propars2)

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
        return get_proatom_rho(self, iatom, propars=propars)

    @property
    def maxiter(self):
        """The maximum iterations."""
        return self._maxiter

    @property
    def threshold(self):
        """The convergence threshold."""
        return self._threshold

    @property
    def radius_cutoff(self):
        """The cutoff radius for local grid."""
        return self._radius_cutoff

    @property
    def mol_pop(self):
        """Molecular population."""
        return self.grid.integrate(self._moldens)

    def _init_log_scheme(self):
        info_list = [
            ("Scheme", "Linear Iterative Stockholder"),
            ("Outer loop convergence threshold", "%.1e" % self.threshold),
            ("Using global ISA", True),
        ]

        info_list.extend(
            [
                ("Maximum outer iterations", self.maxiter),
                ("lmax", self.lmax),
                (
                    "Solver",
                    self._solver.__name__ if callable(self._solver) else self._solver.upper(),
                ),
                ("Basis function type", self._func_type),
                ("Local grid radius", self.radius_cutoff),
            ]
        )
        for k, v in self._solver_options.items():
            info_list.append((k, str(v)))
        deflist(self.logger, info_list)
        self.logger.info(" ")
        # biblio.cite(
        #     "Benda2022", "the use of Linear Iterative Stockholder partitioning"
        # )

    def _finalize_propars(self):
        self._cache.load("charges")

    def _init_propars(self):
        """Initial pro-atom parameters and cache lists."""
        return gisa.init_propars(self)

    @just_once
    def _evaluate_basis_functions(self):
        gisa.evaluate_basis_functions(self)

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
            new_propars = self._opt_propars(**self._solver_options)
            t1 = time.time()
            self.history_time_update_propars_atoms.append(t1 - t0)

            # self.logger.info(f"Time usage for partitioning: {t1-t0:.2f} s")
            propars = self.cache.load("propars")
            propars[:] = new_propars

            t0 = time.time()
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
                pseudo_population = atgrid.rgrid.integrate(4 * np.pi * r**2 * spherical_average)
                charges[iatom] = self.pseudo_numbers[iatom] - pseudo_population
            t1 = time.time()
            self.history_time_update_at_weights.append(t1 - t0)

            self._finalize_propars()
            self.cache.dump("niter", np.nan, tags="o")
            self.cache.dump("change", np.nan, tags="o")

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
                # Note: evaluate pro_shell on a local grid.
                g_ai = self.cache.load(
                    "pro_shell", iatom, ishell, alloc=len(self.radial_distances[iatom])
                )[0]
                g_ai[:] = self.bs_helper.compute_proshell_dens(
                    number, ishell, 1.0, self.radial_distances[iatom], 0
                )
                pro_shells[index, self.local_grids[iatom].indices] = g_ai
                index += 1

        rho_g = self.cache.load("rho*pro_shells", alloc=(nshell, self.grid.size))[0]
        rho_g[:] = self._moldens[None, :] * pro_shells[:, :]

    def calc_promol_dens(self, propars):
        """Compute pro-molecule density based on pro-atom parameters."""
        return np.einsum("np,n->p", self.pro_shells, propars)

    def _opt_propars(self, *args, **kwargs):
        """Optimize pro-atom parameters."""
        if callable(self._solver):
            return self._solver(*args, **kwargs)
        elif isinstance(self._solver, str):
            solver_name = f"solver_{self._solver.replace('-', '_')}"
            if hasattr(self, solver_name):
                return getattr(self, solver_name)(*args, **kwargs)
            else:
                raise RuntimeError(f"Unknown solver: {solver_name}")
        else:
            raise TypeError(f"The type of solver {type(self._solver)} is not supported.")

    @property
    def pro_shells(self):
        """All basis function values on the molecular grid.

        It has a shape of (M, N) where `M` is the total number of basis functions and `N` is the number of
        grid points.
        """
        self.eval_pro_shells()
        return self.cache.load("pro_shells")

    @property
    def pro_shell_centers(self):
        """The coordinates of the center for each basis function.

        It has a shape of (N, 3) where `N` is the total number of basis functions.
        """
        self.eval_pro_shells()
        return self.cache.load("pro_shell_centers")

    @property
    def rho_x_pro_shells(self):
        r"""The intermediate quantity: :math:`\rho(r) \times g_{ak}`.

        It has a shape of (M, N) where `M` is the total number of basis functions and `N` is the number of
        grid points.
        """
        self.eval_pro_shells()
        return self.cache.load("rho*pro_shells")

    def load_pro_shell(self, iatom, ishell):
        """Load one set of basis function values on a local grid.

        Parameters
        ----------
        iatom: int
            The index of the atom in the molecule.
        ishell: int
            The index of the basis function for atom `iatom`.

        Returns
        -------
        np.array
            The basis function values on the local grid with a shape of (N, ), where `N` is the number of points
            in the local grid.
        """
        self.eval_pro_shells()
        return self.cache.load(("pro_shell", iatom, ishell))

    @property
    def propars(self):
        """Load all pro-atom parameters.

        It has a shape of (M, ) where `M` is the total nuber of basis functions.
        """
        return self.cache.load("propars")

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

        gradient = np.zeros((nb_par,))
        hessian = np.zeros((nb_par, nb_par))

        for i in range(nb_par):
            with np.errstate(all="ignore"):
                df_integrand = self.rho_x_pro_shells[i, :] / rho0
            df_integrand[sick] = 0.0
            gradient[i] = -self.grid.integrate(df_integrand)

            if nderiv > 1:
                for j in range(i, nb_par):
                    # Only compute Hessian matrix on a local grid not on full molecular grid.
                    if (
                        np.linalg.norm(self.pro_shell_centers[i] - self.pro_shell_centers[j])
                        > self.radius_cutoff
                    ):
                        hessian[i, j] = 0
                    else:
                        with np.errstate(all="ignore"):
                            hess_integrand = df_integrand * self.pro_shells[j, :] / rho0
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

    def _finalize_propars(self):
        """Restore the pro-atom parameters."""
        iterstock.finalize_propars(self)

    # -----------------------
    # builtin global solvers
    # -----------------------
    def solver_cvxopt(self, allow_neg_pars=False, verbose=False, **cvxopt_options):
        """
        Convex optimization solver.

        Parameters
        ----------
        allow_neg_pars : bool, optional
            Whether negative parameters are allowed.
        verbose : bool, optional
            Whether print more info.
        cvxopt_options : dict
            Settings for ``cvxopt` solver.

        Returns
        -------
        np.array
            Optimized parameters, 1D array.

        """
        rho, propars = self._moldens, self.propars

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
            """The objective function of the global optimization method."""
            if x is None:
                return 0, cvxopt.matrix(propars[:])
            x = np.asarray(x).flatten()
            rho0 = self.calc_promol_dens(x)
            # Note: the propars and pro-mol density could be negative during the optimization.

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
            verbose=verbose,
            # options={"show_progress": log.do_medium, "feastol": self._threshold},
            # options={"show_progress": 3, "feastol": self._threshold},
            options=cvxopt_options,
        )

        propars[:] = np.asarray(opt_CVX["x"]).flatten()
        check_pro_atom_parameters(
            propars,
            pro_atom_density=self.calc_promol_dens(propars),
            total_population=mol_pop,
            check_monotonicity=False,
        )
        return propars

    def solver_sc(self, density_cutoff=1e-15, niter_print=1):
        """
        Self-Consistent solver.

        Parameters
        ----------
        density_cutoff : float, optional
            Density that is smaller than cutoff is set to zero.
        niter_print : int, optional
            Print info every `niter_print` iterations.

        Returns
        -------
        np.array
            Optimized parameters, 1D array.

        """
        propars = self.propars
        self.logger.info("Iteration       Change")
        iter = 0
        while True:
            old_propars = propars.copy()
            propars[:] = self.function_g(propars, density_cutoff=density_cutoff)
            change = self.compute_change(propars, old_propars)
            self.logger.debug(f"            {iter+1:<4}    {change:.3e}")
            if iter % niter_print == 0:
                self.logger.info("%9i   %10.5e" % (iter, change))
            if change < self._threshold:
                break
            iter += 1
        return propars

    def function_g(self, x, density_cutoff=1e-15):
        """The fixed-point equation :math:`g(x)=x`."""
        # 1. load molecular and pro-molecule density from cache
        rho = self._moldens
        old_rho0 = self.calc_promol_dens(x)
        sick = (rho < density_cutoff) | (old_rho0 < density_cutoff)

        new_x = np.zeros_like(x)
        ishell = 0
        for iatom in range(self.natom):
            # 2. load old propars
            x_iatom = x[self._ranges[iatom] : self._ranges[iatom + 1]]

            # 3. compute basis functions on molecule grid
            fun_val = []
            local_grid = self.local_grids[iatom]
            indices = local_grid.indices
            for k, c_ak in enumerate(x_iatom.copy()):
                g_ak = self.load_pro_shell(iatom, k)
                rho0_ak = g_ak * c_ak

                with np.errstate(all="ignore"):
                    integrand = rho[indices] * rho0_ak / old_rho0[indices]
                integrand[sick[indices]] = 0.0

                fun_val.append(local_grid.integrate(integrand))
                ishell += 1

            # 4. set new x values
            new_x[self._ranges[iatom] : self._ranges[iatom + 1]] = np.asarray(fun_val)
        return new_x

    # TODO: this is duplicated
    # def solver_diis(self, maxiter=1000, diis_size=8, use_dmrs=False, version="P"):
    def solver_diis(self, use_dmrs=False, **diis_options):
        """DIIS solver"""

        def conv_func(residual, x, old_x):
            if use_dmrs:
                val = np.linalg.norm(residual)
            else:
                old_rho0 = self.calc_promol_dens(old_x)
                rho0 = self.calc_promol_dens(x)
                val = np.sqrt(self.grid.integrate((rho0 - old_rho0) ** 2))
            return val

        propars = self.propars
        propars[:] = diis(
            propars,
            self.function_g,
            self.threshold,
            conv_func=conv_func,
            verbose=True,
            logger=self.logger,
            **diis_options,
        )

        check_pro_atom_parameters(
            propars,
            pro_atom_density=self.calc_promol_dens(propars),
            total_population=self.mol_pop,
            check_monotonicity=False,
        )
        return propars

    def solver_trust_region(self, allow_neg_pars=False):
        """Optimize the promodel using the trust-constr minimizer from SciPy."""
        rho, pars0 = self._moldens, self.propars

        # Compute the total population
        pop = self.grid.integrate(rho)
        self.logger.info("Integral of density: {pop}")
        nb_par = len(pars0)

        def cost_grad(x):
            rho0 = self.calc_promol_dens(x)
            f, df = self._working_matrix(rho, rho0, nb_par, 1)
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
            rho0 = self.calc_promol_dens(_current_pars)
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
        self.logger.info(f'Optimizer message: "{optresult.message}"')
        if not optresult.success:
            raise RuntimeError("Convergence failure.")

        rho0 = self.calc_promol_dens(optresult.x)
        constrain = self.grid.integrate(rho0) - pop
        self.logger.info(f"Constraint: {constrain}")
        self.logger.info("Optimized parameters: ")
        self.logger.info(optresult.x)
        return optresult.x

    def residual(self, x):
        """The definition of residual."""
        return self.function_g(x) - x

    # TODO: use **solver_options instead of passing all parameters.
    def solver_cdiis(self, **cdiis_options):
        conv, nbiter, rnormlist, mklist, cnormlist, xlast = cdiis(
            self.propars,
            self.function_g,
            self.threshold,
            self.maxiter,
            logger=self.logger,
            verbose=True,
            **cdiis_options,
        )
        if not conv:
            raise RuntimeError("Not converged!")
        check_pro_atom_parameters(xlast)
        return xlast
