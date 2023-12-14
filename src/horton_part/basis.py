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


import numpy as np
from scipy.special import gamma
import json
from importlib_resources import files

__all__ = [
    "BasisFuncHelper",
]

# Constants
DEFAULT_NB_EXP = 6
JSON_DATA_PATH = files("horton_part.data")


def _load_params(func_type):
    filename = JSON_DATA_PATH.joinpath(f"{func_type}.json")
    with open(filename) as file:
        data = json.load(file)
    orders, exps, inits = {}, {}, {}
    for number, data in data.items():
        number = int(number)
        orders[number] = data[0]
        exps[number] = data[1]
        inits[number] = data[2]
    return orders, exps, inits


def _generate_default_initials(func_type, number, nshell):
    if func_type == "slater":
        return np.array(
            [2 * number ** (1 - (i - 1) / (nshell - 1)) for i in range(1, nshell + 1)]
        )
    else:
        return np.ones(nshell, float) * number / nshell


def evaluate_function(n, population, alpha, r, nderiv=0, axis=None):
    r"""
    Evaluate general function and its derivative, vectorized for n, population, and alpha.

    .. math::

        f(\mathbf{r}) = N(\alpha, n) \exp^{-\alpha |\mathbf{r}|^n}

        \int N(\alpha, n)  f(\mathbf{r}) d\mathbf{r} =  4 \pi * \int_0^{\infty} r^2  \exp^{-\alpha r^n} = 1

        N(\alpha, n) = \frac{n \alpha^{3/n}}{4 \pi \Gamma(3/n)}

        N(\alpha, 1) = \frac{\alpha^3}{8 \pi}

        N(\alpha, 2) = (\frac{\alpha}{\pi})^{3/2}

        \frac{\partial{f(\mathbf{r})}}{\partial{r}} = - n r^{n-1} \alpha f(\mathbf{r})

     Parameters
    ----------
    n : float or np.ndarray
        Order, positive. Scalar or 1D array.
    population : float
        Constant representing population.
    alpha : float or np.ndarray
        Exponential coefficient. Scalar or 1D array matching n.
    r : np.ndarray
        Points. 1D array.
    nderiv : int, optional
        The order of derivative. Currently supports 0 (function) and 1 (first derivative).
    axis : int, optional
        The axis of the array to sum. Applicable if n, alpha are arrays.

    Returns
    -------
    f : np.ndarray
        Function values at points r.
    df : np.ndarray, optional
        Derivative of function at points r, if nderiv == 1.

    """
    if np.any(n <= 0):
        raise ValueError("n must be positive")
    if np.any(alpha < 0):
        raise ValueError("alpha must be non-negative")
    if not isinstance(r, np.ndarray):
        raise ValueError("r must be a numpy array")

    prefactor = population * n * alpha ** (3 / n) / (4 * np.pi * gamma(3 / n))
    if np.isscalar(n):
        assert np.isscalar(population) and np.isscalar(alpha)
        f = prefactor * np.exp(-alpha * r**n)
        if nderiv > 0:
            df = -n * r ** (n - 1) * f
    else:
        f = prefactor[:, np.newaxis] * np.exp(
            -alpha[:, np.newaxis] * r[np.newaxis, :] ** n[:, np.newaxis]
        )

        if nderiv > 0:
            df = -n[:, np.newaxis] * r[np.newaxis, :] ** (n[:, np.newaxis] - 1) * f

        if axis is not None:
            f = np.sum(f, axis=axis)
            if nderiv > 0:
                df = np.sum(df, axis=axis)

    if nderiv == 0:
        return f
    elif nderiv == 1:
        return f, df
    else:
        raise NotImplementedError


class BasisFuncHelper:
    """Helper class for basis function."""

    def __init__(self, exponents_orders, exponents=None, initials=None):
        self._orders = exponents_orders
        self._exponents = exponents
        self._initials = initials
        self._cache = {}

    def get_nshell(self, number):
        """Get number of basis functions based on the atomic number of an atom."""
        return len(self._exponents[number])

    def get_order(self, number, ishell=None):
        """Get exponent order for an atom for its `ishell` basis function."""
        return self._orders[number] if ishell is None else self._orders[number][ishell]

    def get_initial(self, number, ishell=None):
        """Get initial value for an atom for its `ishell` basis function."""
        return (
            self._initials[number] if ishell is None else self._initials[number][ishell]
        )

    def get_exponent(self, number, ishell=None):
        """Get exponent coefficient for an atom for its `ishell` basis function."""
        return (
            self._exponents[number]
            if ishell is None
            else self._exponents[number][ishell]
        )

    def compute_proshell_dens(self, number, ishell, population, points, nderiv=0):
        """Compute pro-shell density on points for an atom."""
        order = self.get_order(number, ishell)
        exp = self.get_exponent(number, ishell)
        return evaluate_function(order, population, exp, points, nderiv, axis=0)

    # def compute_proatom_dens_slow(self, number, populations, points, nderiv=0, axis=0):
    #     """Compute pro-atom density on points for an atom."""
    #     orders = self.get_order(number)
    #     exps = self.get_exponent(number)
    #     return evaluate_function(orders, populations, exps, points, nderiv, axis=axis)

    def compute_proatom_dens(self, number, populations, points, nderiv=0):
        """Compute pro-atom density on points for an atom."""
        self.get_exponent(number)
        nshell = self.get_nshell(number)
        y = d = 0.0
        for i in range(nshell):
            res = self.compute_proshell_dens(number, i, populations[i], points, nderiv)
            if nderiv == 0:
                y += res
            elif nderiv == 1:
                y += res[0]
                d += res[1]
            else:
                raise NotImplementedError

        if nderiv == 0:
            return y
        elif nderiv == 1:
            return y, d
        else:
            raise RuntimeError("nderiv should only be 0 or 1.")

    @classmethod
    def from_type(cls, func_type="gauss"):
        """Construct class from basis type."""
        orders, exponents, initials = _load_params(func_type)
        for number, exps in exponents.items():
            if number not in initials:
                initials[number] = _generate_default_initials(
                    func_type, number, len(exps)
                )
        return cls(orders, exponents, initials)
