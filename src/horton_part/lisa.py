# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
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
from .wrapper import log
from .gisa import GaussianIterativeStockholderWPart, get_pro_a_k


__all__ = ["LinearIterativeStockholderWPart"]


class LinearIterativeStockholderWPart(GaussianIterativeStockholderWPart):
    name = "lisa"

    def _init_log_scheme(self):
        if log.do_medium:
            log.deflist(
                [
                    ("Scheme", "Linear Iterative Stockholder"),
                    ("Convergence threshold", "%.1e" % self._threshold),
                    ("Maximum iterations", self._maxiter),
                ]
            )
            # biblio.cite(
            #     "Robert2022 the use of Linear Iterative Stockholder ", "partitioning"
            # )

    def _opt_propars(self, rho, propars, rgrid, alphas, threshold):
        if self._obj_fn_type == 1:
            return self._opt_propars_with_lisa_method(
                rho, propars, rgrid, alphas, threshold
            )
        elif self._obj_fn_type == 2:
            return self._opt_propars_with_mbis_lagrangian(
                rho, propars, rgrid, alphas, threshold
            )
        else:
            raise NotImplementedError

    @staticmethod
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
        r = rgrid.radii
        oldF = None
        for irep in range(1000):
            # compute the contributions to the pro-atom
            terms = np.array(
                [get_pro_a_k(propars[k], alphas[k], r) for k in range(nprim)]
            )
            pro = terms.sum(axis=0)
            newF = -rgrid.integrate(r**2 * rho * np.log(pro))
            # transform to partitions
            terms *= rho / pro
            # the partitions and the updated parameters
            for k in range(nprim):
                propars[k] = rgrid.integrate(terms[k])
            # check for convergence
            if oldF is None:
                change = 1e100
            else:
                change = np.abs(oldF - newF)
            if change < threshold:
                return propars
            oldF = newF
        log("Not converge, but go ahead!")
        # The initial values could lead to converged issues.
        # assert False
        return propars

    @staticmethod
    def _opt_propars_with_lisa_method(
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

        ###########################
        # Linear equality constraints :
        # Ax = b with x=(c_(a,k))_{k=1..Ka} ; A = (1...1) and b = Na = (Na)
        matrix_constraint_eq = cvxopt.matrix(1.0, (1, nprim))

        r = rgrid.radii
        # N_a : corresponds to the EQUALITY constraint sum_{k=1..Ka} c_(a,k) = N_a
        N_a = rgrid.integrate(rho)
        vector_constraint_eq = cvxopt.matrix(N_a, (1, 1))

        # Use optimized x to calculate Gaussian functions
        gauss_funcs = np.array([get_pro_a_k(1.0, alphas[k], r) for k in range(nprim)])

        def F(x=None, z=None):
            # x is the optimized coefficients
            if x is None:
                # For the initial step, this should be propars
                return 0, cvxopt.matrix(propars[:])

            x = np.clip(x, 1e-6, None)  # Replace values < 1e-6 with 1e-6

            # Use optimized to calculate density from each Gaussian function.
            gauss_pros = np.array(
                [get_pro_a_k(x[k], alphas[k], r) for k in range(nprim)]
            )
            pro = gauss_pros.sum(axis=0)

            f = -rgrid.integrate(rho * np.log(pro))
            df = np.zeros((1, nprim), float)
            for i in range(nprim):
                # NOTE: in Horton's grid, the 4 * \pi * r**2 is already included
                df[0, i] = -rgrid.integrate(rho * gauss_funcs[i] / pro)
            df = cvxopt.matrix(df)

            if verbose:
                print("f=", f)
                print("df=", df)

            if z is None:
                return f, df

            hess = np.zeros((nprim, nprim), float)
            for i in range(nprim):
                for j in range(i, nprim):
                    hess[i, j] = rgrid.integrate(
                        # NOTE: in Horton's grid, the 4 * \pi * r**2 is already included
                        rho
                        * gauss_funcs[i]
                        * gauss_funcs[j]
                        / pro**2
                    )
                    hess[j, i] = hess[i, j]
            hess = z[0] * cvxopt.matrix(hess)

            if verbose:
                print("hess:")
                print(hess)

            return f, df, hess

        opt_CVX = cvxopt.solvers.cp(
            F,
            G=matrix_constraint_ineq,
            h=vector_constraint_ineq,
            A=matrix_constraint_eq,
            b=vector_constraint_eq,
            verbose=verbose,
            reltol=threshold,
        )

        optimized_res = opt_CVX["x"]
        assert (np.asarray(optimized_res) > 0).all()
        assert np.sum(optimized_res) - N_a < 1e-8

        new_propars = np.asarray(opt_CVX["x"]).flatten()
        return new_propars
