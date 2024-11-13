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
from scipy.linalg import solve
from scipy.optimize import SR1, LinearConstraint, minimize

from horton_part import gisa
from horton_part.core import iterstock

from .algo import bfgs, cdiis, diis
from .alisa import setup_bs_helper
from .core.cache import just_once
from .core.iterstock import compute_change
from .core.logging import deflist
from .core.stockholder import AbstractStockholderWPart
from .gisa import get_proatom_rho
from .utils import NEGATIVE_CUTOFF, check_pro_atom_parameters, fix_propars

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
        # radius_cutoff=np.inf,
        solver="cvxopt",
        solver_options=None,
        basis_func="gauss",
        grid_type=2,
        **kwargs,
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
        # self._radius_cutoff = radius_cutoff
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
        self._ranges = []

        super().__init__(
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            lmax,
            logger,
            grid_type,
        )

    def setup_grids(self):
        assert self.grid_type in [1, 2, 3]
        self._on_molgrid = True if self.grid_type in [2, 3] else False
        self._only_use_molgrid = True if self.grid_type in [3] else False

    # -------------------------
    # Turn off local grids info
    # -------------------------
    def get_rgrid(self, index):
        """Load radial grid."""
        if self.only_use_molgrid:
            self.logger.debug("rgird is not available when only_use_molgrid is `True`.")
            raise NotImplementedError
        else:
            return self.get_grid(index).rgrid

    def to_atomic_grid(self, index, data):
        if self.only_use_molgrid:
            self.logger.debug("atom grids are not available when only_use_molgrid is `True`.")
            raise NotImplementedError
        else:
            return super().to_atomic_grid(index, data)

    def get_proatom_rho(self, iatom, propars=None, **kwargs):
        """Get pro-atom density for atom `iatom`.

        If `propars` is `None`, the cache values are used; otherwise, the `propars` are used.

        Parameters
        ----------
        iatom: int
            The index of atom `iatom`.
        propars: 1D np.array
            The pro-atom parameters.

        """
        return get_proatom_rho(self, iatom, propars=propars)

    # -------------------------

    @property
    def bs_helper(self):
        """A basis function helper."""
        return setup_bs_helper(self)

    def compute_change(self, propars1, propars2):
        """Compute the difference between an old and a new proatoms"""
        return compute_change(self, propars1, propars2)

    @property
    def maxiter(self):
        """The maximum iterations."""
        return self._maxiter

    @property
    def threshold(self):
        """The convergence threshold."""
        return self._threshold

    # @property
    # def radius_cutoff(self):
    #     """The cutoff radius for local grid."""
    #     return self._radius_cutoff

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
                # ("Local grid radius", self.radius_cutoff),
            ]
        )
        for k, v in self._solver_options.items():
            info_list.append((k, str(v)))
        deflist(self.logger, info_list)
        self.logger.info(" ")
        # biblio.cite(
        #     "Benda2022", "the use of Linear Iterative Stockholder partitioning"
        # )

    def _init_propars(self):
        """Initial pro-atom parameters and cache lists."""
        self.logger.debug("Initial propars ...")
        propars = gisa.init_propars(self)
        return propars

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

    @just_once
    def do_partitioning(self):
        """Do global partitioning scheme."""
        new = any(f"at_weights_{i}" not in self.cache for i in range(self.natom))
        new |= "niter" not in self.cache
        if new:
            # Initialize pro-atom parameters.
            self._init_propars()
            # Evaluate basis functions on molecular grid.
            # Note, here on_molgrid is True always even for grid_type = 1 and 2
            gisa.evaluate_basis_functions(self, force_on_molgrid=True)

            t0 = time.time()
            new_propars = self._opt_propars(**self._solver_options)
            t1 = time.time()
            self.history_time_update_propars.append(t1 - t0)

            # self.logger.info(f"Time usage for partitioning: {t1-t0:.2f} s")
            propars = self.cache.load("propars")
            propars[:] = new_propars

            t0 = time.time()
            self.update_at_weights(force_on_molgrid=True)
            # compute the new charge
            charges = self.cache.load("charges", alloc=self.natom, tags="o")[0]
            for iatom in range(self.natom):
                at_weights = self.cache.load(f"at_weights_{iatom}")
                nelec = self.grid.integrate(at_weights * self._moldens)
                charge = self.pseudo_numbers[iatom] - nelec
                charges[iatom] = charge
            t1 = time.time()

            self.history_time_update_at_weights.append(t1 - t0)
            self._finalize_propars()

    def is_promol_valid(self, propars):
        """Check if the promol density is valid."""
        valid = True
        for iatom in range(self.natom):
            valid_i = self.is_proatom_valid(iatom, propars)
            if valid_i != 0:
                valid = False
                break
        return valid

    def is_proatom_valid(self, iatom, propars):
        """Check if the proatom density is valid."""
        rho0_iatom, _ = get_proatom_rho(self, iatom, propars)
        valid = 0
        if (rho0_iatom < NEGATIVE_CUTOFF).any() or (
            rho0_iatom[:-1] - rho0_iatom[1:] < NEGATIVE_CUTOFF
        ).any():
            if (rho0_iatom < NEGATIVE_CUTOFF).any():
                valid = 1
            else:
                valid = 2
        return valid

    # def check_pro(self, iatom, propars):
    #     """Check if the proatom paramters are valid."""
    #     valid = self.is_proatom_valid(iatom, propars)
    #     if valid != 0:
    #         self.logger.debug(
    #             f"{PERIODIC_TABLE[self.numbers[iatom]]} with index = {iatom}"
    #         )
    #         exps = self.bs_helper.exponents[self.numbers[iatom]]
    #         caks = propars[self._ranges[iatom] : self._ranges[iatom + 1]]
    #         self.logger.debug(f"{'Exp.':>20}{'c_ak':>20} ")
    #         for exp, c_ak in zip(exps, caks):
    #             self.logger.debug(f"{exp:>20.8f}       {c_ak:>20.8f}")
    #         if valid == 1:
    #             raise RuntimeError("The proatom density is negative somewhere!")
    #         elif valid == 2:
    #             raise RuntimeError("The proatom density is not monotonic decay!")

    @just_once
    def eval_pro_shells(self):
        """Evaluate pro-shell functions on molecular grids."""
        nshell = len(self.cache.load("propars"))
        pro_shells = self.cache.load("pro_shells", alloc=(nshell, self.grid.size))[0]
        for iatom in range(self.natom):
            bs_funcs_i = self.cache.load(f"bs_funcs_{iatom}")
            pro_shells[self._ranges[iatom] : self._ranges[iatom + 1], :] = bs_funcs_i
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
        return self.cache.load(f"bs_funcs_{iatom}")[ishell]
        # return self.cache.load(("pro_shell", iatom, ishell))

    @property
    def propars(self):
        """Load all pro-atom parameters.

        It has a shape of (M, ) where `M` is the total nuber of basis functions.
        """
        return self.cache.load("propars")

    def _working_matrix(self, rho, rho0, nb_par, nderiv=0):
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
        sick = (rho < self.density_cutoff) | (rho0 < self.density_cutoff)
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
        history_propars = [propars.copy()]

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
            if not np.allclose(x, history_propars[-1]):
                history_propars.append(x)
            return f, cvxopt.matrix(df), hess

        opt_CVX = cvxopt.solvers.cp(
            obj_func,
            G=matrix_constraint_ineq,
            h=vector_constraint_ineq,
            A=matrix_constraint_eq,
            b=vector_constraint_eq,
            verbose=verbose,
            options=cvxopt_options,
        )

        propars[:] = np.asarray(opt_CVX["x"]).flatten()
        check_pro_atom_parameters(
            propars,
            pro_atom_density=self.calc_promol_dens(propars),
            total_population=mol_pop,
            check_monotonicity=False,
            logger=self.logger,
        )

        # TODO: collect info
        if opt_CVX["status"] == "optimal":
            self.cache.dump("niter", len(history_propars) - 1, tags="o")
            for x in history_propars[:-1]:
                rho0 = self.calc_promol_dens(x)
                entropy = self._compute_entropy(self._moldens, rho0)
                self.history_entropies.append(entropy)
            self.history_propars = history_propars[1:]
            return propars
        else:
            raise RuntimeError("CVXOPT not converged!")

    def solver_newton(self, maxiter=100):
        """Default, Exact Newton method."""
        return self._solver_general_newton(maxiter=maxiter, mode="exact")

    def solver_m_newton(
        self,
        maxiter=100,
        linspace_size=40,
        tau=1.0,
        linesearch_mode="valid-promol",
        density_cutoff=1e-15,
    ):
        """Default Modified Newton method"""
        return self._solver_general_newton(
            mode="modified",
            maxiter=maxiter,
            linesearch_mode=linesearch_mode,
            tau=tau,
            linspace_size=linspace_size,
        )

    def solver_quasi_newton(
        self,
        mode="bfgs",
        maxiter=1000,
        niter_exact_newton=0,
        linesearch_mode="valid-promol",
        tau=1.0,
        linspace_size=40,
        density_cutoff=1e-15,
    ):
        """Default Quasi-Newton method."""
        assert mode in ["bfgs"]
        return self._solver_general_newton(
            mode=mode,
            maxiter=maxiter,
            niter_exact_newton=niter_exact_newton,
            linesearch_mode=linesearch_mode,
            tau=tau,
            linspace_size=linspace_size,
        )

    def _solver_general_newton(
        self,
        mode="bfgs",
        maxiter=1000,
        niter_exact_newton=0,
        linesearch_mode="valid-promol",
        tau=1.0,
        linspace_size=40,
    ):
        """
        Implements a Newton solver with various modes and line search strategies for optimization problems,
        particularly in molecular density calculations.

        Parameters
        ----------
        mode : str, optional
            Mode for the Newton solver. Choices are 'exact', 'modified', and 'bfgs'. Default is 'bfgs'.
        maxiter : int, optional
            Maximum number of iterations for the Newton solver. Default is 1000.
        niter_exact_newton : int, optional
            Number of iterations to use Newton's method. Default is 0.
        linesearch_mode : str, optional
            Mode for the line search algorithm. Choices are 'valid-promol' and 'with-extended-kl'.
            Default is 'valid-promol'.
        tau : float, optional
            Initial step size in the line search. Default is 1.0.
        linspace_size : int, optional
            Size of the linspace used in the line search. Default is 40.

        Raises
        ------
        RuntimeError
            If the provided mode or linesearch_mode is invalid.
            If the line search fails.
            If the solver does not converge within the maximum iterations.

        Returns
        -------
        list
            Updated parameters (propars) after optimization.

        """
        valid_modes = ["exact", "modified", "bfgs"]
        if mode not in valid_modes:
            raise RuntimeError(f"Wrong Newton mode :{mode}. It should be one of {valid_modes}")

        valid_ls_modes = ["valid-promol", "with-extended-kl"]
        if linesearch_mode not in valid_ls_modes:
            raise RuntimeError(
                f"Wrong linesearch_mode {linesearch_mode}. It should be one of {valid_ls_modes}"
            )

        assert tau >= 0
        rho, propars = self._moldens, self.propars
        pop = self.grid.integrate(rho)
        nb_par = len(propars)

        def linesearch(mode, delta, propars, old_pro_pop=None, old_f=None):
            if mode == "exact":
                return propars + delta, delta

            # check entropy
            self.logger.debug(f"Extended KL: {old_f+old_pro_pop:.8f}, KL: {old_f:.8f}")

            # We have a set of Gaussian functions with a set of exponential coefficients defined in an array
            # called exp_array. The prefactors of these functions are stored in another array, named propars.
            # These prefactors are updated as follows: propars = propars + delta, where delta is an increment array.
            # The issue arises when the prefactor of the most diffused basis function, corresponding to the smallest
            # exponential coefficient, cannot be negative. Therefore, we need to find the position of the most diffused
            # function and determine the corresponding propars. If the corresponding c_ak is already zero, we need to
            # check the second most diffused function, and so on, until we find the first non-zero one, which must
            # always be positive.
            # We then record all indices of the diffused functions we checked and the corresponding c_ak being zero.
            # For these propars, we need to check if the corresponding delta elements are negative or non-negative.
            # If they are positive, we can safely update propars = propars + delta; otherwise, we retain zeros.

            global_fixed_index = np.ones_like(delta)
            for iatom in range(self.natom):
                exp_array_i = self.bs_helper.get_exponent(self.numbers[iatom])
                propars_i = propars[self._ranges[iatom] : self._ranges[iatom + 1]]
                delta_i = delta[self._ranges[iatom] : self._ranges[iatom + 1]]
                fixed_index = fix_propars(exp_array_i, propars_i, delta_i)
                for _i in fixed_index:
                    global_fixed_index[_i + self._ranges[iatom]] = 0
            self.logger.debug(f"global_fixed_index: {global_fixed_index}")

            x, _s, new_propars = None, None, None
            for x in np.linspace(tau, 0, linspace_size, endpoint=False):
                _s = x * delta
                new_propars = propars + global_fixed_index * _s
                # new_propars *= global_fixed_index

                if self.is_promol_valid(new_propars):
                    if linesearch_mode == "valid-promol":
                        break
                    elif linesearch_mode == "with-extended-kl":
                        new_pro = self.calc_promol_dens(new_propars)
                        f = self._working_matrix(rho, new_pro, nb_par, 0)
                        pro_pop = self.grid.integrate(new_pro)
                        extended_kl = f + pro_pop
                        old_extended_kl = old_f + old_pro_pop

                        if (extended_kl - old_extended_kl) < NEGATIVE_CUTOFF:
                            break
            else:
                # self-consistent step
                # new_propars = self.function_g(propars, density_cutoff=density_cutoff)
                # _s = 0.5 * delta
                # TODO: the line search may fail when c_ak of the most diffused function is negative.
                self.logger.debug(f"delta: {delta}")
                self.logger.debug(f"propars: {propars}")
                self.logger.debug(f"new_propars: {new_propars}")
                raise RuntimeError("Line search failed!")

            self.logger.debug(f"x: {x}")
            self.logger.debug(f" sum of delta: {np.sum(delta)}")
            self.logger.debug(f" sum of propars: {np.sum(new_propars)}")
            return new_propars, _s

        H = None
        olddf = None
        oldH = None
        s = None
        self.logger.info("            Iter.    Change    Entropy")
        self.logger.info("            -----    ------    -------")
        for irep in range(maxiter):
            old_propars = propars.copy()
            pro = self.calc_promol_dens(propars)

            if mode == "bfgs":
                if irep == 0 or irep <= niter_exact_newton - 1:
                    if niter_exact_newton == 0:
                        f, df = self._working_matrix(rho, pro, nb_par, 1)
                        hess = np.identity(len(df))
                    else:
                        f, df, hess = self._working_matrix(rho, pro, nb_par, 2)
                    H = np.linalg.inv(hess)
                else:
                    f, df = self._working_matrix(rho, pro, nb_par, 1)
                    H = bfgs(df, s, olddf, oldH)

                self.logger.debug("Hessian matrix:")
                self.logger.debug(H)
                delta = H @ (-1 - df)
            elif mode in ["exact", "modified"]:
                f, df, hess = self._working_matrix(rho, pro, nb_par, 2)
                try:
                    delta = solve(hess, -1 - df, assume_a="sym")
                except np.linalg.LinAlgError as e:
                    raise RuntimeError(e)
            else:
                raise NotImplementedError

            propars[:], s = linesearch(
                mode, delta, propars, old_pro_pop=self.grid.integrate(pro), old_f=f
            )

            # Note: the first entropy corresponds to the initial values.
            # Compute entropy, and we use oldpro instead rho to be consistent with LISA method.
            entropy = self._compute_entropy(rho, pro)
            change = self.compute_change(propars, old_propars)
            self.history_entropies.append(entropy)
            self.history_propars.append(propars.copy())
            self.history_changes.append(change)

            self.logger.info(f"            {irep+1:<4}    {change:.5e}    {entropy:.5e}")

            if change < self.threshold:
                check_pro_atom_parameters(
                    propars,
                    total_population=pop,
                    pro_atom_density=pro,
                    check_monotonicity=False,
                    check_propars_negativity=False,
                    logger=self.logger,
                )
                self.cache.dump("niter", irep + 1, tags="o")
                return propars

            olddf = df
            oldH = H
        else:
            raise RuntimeError("Not converged!")

    def solver_sc(self, niter_print=1):
        """
        Self-Consistent solver.

        Parameters
        ----------
        niter_print : int, optional
            Print info every `niter_print` iterations.

        Returns
        -------
        np.array
            Optimized parameters, 1D array.

        """
        propars = self.propars
        self.logger.info("Iteration       Change      Entropy")
        iter = 0
        while True:
            old_propars = propars.copy()

            propars[:] = self.function_g(propars)
            self.logger.debug(propars)
            change = self.compute_change(propars, old_propars)

            # Be consistent to LISA entropy corresponding to old_propars
            rho0 = self.calc_promol_dens(old_propars)
            entropy = self._compute_entropy(self._moldens, rho0)
            self.history_entropies.append(entropy)
            self.history_changes.append(change)
            self.history_propars.append(propars.copy())

            # self.logger.debug(f"            {iter+1:<4}    {change:.3e}")
            if (iter + 1) % niter_print == 0:
                self.logger.info("%9i   %10.5e   %10.5e" % (iter + 1, change, entropy))

            if change < self._threshold:
                break
            iter += 1

        # TODO: collect opt info
        self.cache.dump("niter", iter + 1, tags="o")
        # self.cache.dump("change", change, tags="o")
        return propars

    def function_g(self, x):
        """The fixed-point equation :math:`g(x)=x`."""
        # 1. load molecular and pro-molecule density from cache
        self.logger.debug("Enter function_g ...")
        rho = self._moldens
        old_rho0 = self.calc_promol_dens(x)
        sick = (rho < self.density_cutoff) | (old_rho0 < self.density_cutoff)

        new_x = np.zeros_like(x)
        ishell = 0
        for iatom in range(self.natom):
            # 2. load old propars
            x_iatom = x[self._ranges[iatom] : self._ranges[iatom + 1]]

            # 3. compute basis functions on molecule grid
            fun_val = []
            for k, c_ak in enumerate(x_iatom.copy()):
                g_ak = self.load_pro_shell(iatom, k)
                rho0_ak = g_ak * c_ak
                with np.errstate(all="ignore"):
                    integrand = rho * rho0_ak / old_rho0
                integrand[sick] = 0.0

                fun_val.append(self.grid.integrate(integrand))
                ishell += 1

            # 4. set new x values
            new_x[self._ranges[iatom] : self._ranges[iatom + 1]] = np.asarray(fun_val)
        self.logger.debug("Out function_g")
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
        propars[:], niter, history_propars = diis(
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
            logger=self.logger,
        )

        # Collect info
        self.cache.dump("niter", niter, tags="o")
        for x in history_propars[1:]:
            rho0 = self.calc_promol_dens(x)
            entropy = self._compute_entropy(self._moldens, rho0)
            self.history_entropies.append(entropy)
        self.history_propars = history_propars[1:]
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
        # TODO: collect opt info
        return optresult.x

    def residual(self, x):
        """The definition of residual."""
        return self.function_g(x) - x

    def solver_cdiis(self, **cdiis_options):
        """CDIIS solver."""
        conv, nbiter, rnormlist, mklist, cnormlist, propars, history_propars = cdiis(
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
        check_pro_atom_parameters(
            propars,
            pro_atom_density=self.calc_promol_dens(propars),
            total_population=self.mol_pop,
            check_monotonicity=False,
            logger=self.logger,
        )

        self.cache.dump("niter", nbiter, tags="o")
        # self.cache.dump("change", rnormlist[-1], tags="o")

        for x in history_propars[1:]:
            rho0 = self.calc_promol_dens(x)
            entropy = self._compute_entropy(self._moldens, rho0)
            self.history_entropies.append(entropy)
        self.history_propars = history_propars[1:]
        self.history_changes = rnormlist
        return propars
