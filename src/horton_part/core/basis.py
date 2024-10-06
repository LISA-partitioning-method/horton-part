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
"""Module for handling basis functions used in GISA and LISA schemes."""

import json

import numpy as np
import yaml
from scipy.special import gamma

from horton_part.utils import DATA_PATH

__all__ = ["BasisFuncHelper", "evaluate_function", "load_params"]


def load_params(filename, extension="json"):
    """
    Loads parameters from a JSON or YAML file.

    The function reads a file specified by the filename, which can be in JSON or YAML format,
    and extracts orders, exponents, and initial values for different elements based on their atomic numbers.

    Parameters
    ----------
    filename : str
        The path to the file containing the parameters in JSON or YAML format.
    extension : str, optional
        The format of the file, either 'json' or 'yaml'. Default is 'json'.

    Returns
    -------
    tuple of dict
        A tuple containing three dictionaries: orders, exponents, and initials.
        Each dictionary maps an atomic number (int) to its corresponding numpy array values (orders, exponents, and initials).
    """
    assert extension.lower() in ["json", "yaml"], "Format must be 'json' or 'yaml'."

    # Load the data based on the format
    if extension.lower() == "json":
        with open(filename) as file:
            data = json.load(file)
    else:
        with open(filename) as file:
            data = yaml.safe_load(file)

    # Initialize the dictionaries to store orders, exponents, and initial values
    orders, exps, inits = {}, {}, {}

    # Process the data and fill the dictionaries
    for number, values in data.items():
        number = int(number)  # Convert the atomic number to an integer
        orders[number] = np.asarray(values[0])
        exps[number] = np.asarray(values[1])
        inits[number] = np.asarray(values[2])

    return orders, exps, inits


def evaluate_function(n, population, alpha, r, nderiv=0, axis=None):
    r"""
    Evaluate a general function and its derivative, with support for vectorized operations.

    This function calculates a custom mathematical function :math:`f(\mathbf{r})` and optionally its derivative,
    given certain parameters. The calculation can be vectorized over multiple values of :math:`n`,
    :math:`\text{population}`, and :math:`\alpha`.

    The function is defined as:

    .. math::

        f(\mathbf{r}) = N(\alpha, n) \exp(-\alpha |\mathbf{r}|^n)

    where :math:`N(\alpha, n)` is a normalization factor ensuring the integral of :math:`f` over all space equals 1.
    The derivative of :math:`f` with respect to :math:`r` is also provided.

    The normalization factor can be obtained as follows:

    .. math::

        \int N(\alpha, n)  f(\mathbf{r}) d\mathbf{r} =  4 \pi * \int_0^{\infty} r^2  \exp^{-\alpha r^n} = 1

        N(\alpha, n) = \frac{n \alpha^{3/n}}{4 \pi \Gamma(3/n)}

    where N(α, n) is a normalization factor ensuring the integral of f over all space equals 1.


    For Slater function, i.e., :math:`n=1`, the normalization factor is

    .. math::

        N(\alpha, 1) = \frac{\alpha^3}{8 \pi}

    For Gaussian function, :math:`n=2`, the normalization factor is

    .. math::

        N(\alpha, 2) = (\frac{\alpha}{\pi})^{3/2}

    The derivative of function f w.r.t the :math:`|r|` is defined as:

    .. math::

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
    use_vec_version = not np.isscalar(n)
    if use_vec_version:
        # TODO: this version is slower than using for loop
        assert not (np.isscalar(alpha) or np.isscalar(population))
        prefactor = prefactor[:, np.newaxis]
        alpha = alpha[:, np.newaxis]
        r = r[np.newaxis, :]
        n = n[:, np.newaxis]

    f = prefactor * np.exp(-alpha * r**n)
    if nderiv > 0:
        df = -n * r ** (n - 1) * f

    # Summing across the specified axis if required
    if use_vec_version and axis is not None:
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
    """
    A helper class for handling basis functions in GISA and LISA methods.

    This class provides functionalities to load and access parameters of basis functions
    used in GISA and LISA methods. It supports operations for different atoms characterized
    by atomic numbers, and allows computations of densities and other properties at
    specified points in space.

    """

    def __init__(self, exponents_orders, exponents=None, initials=None):
        self._orders = exponents_orders
        self._exponents = exponents
        self._initials = initials

    @property
    def orders(self):
        """A dictionary mapping atomic numbers to their exponential orders."""
        return self._orders

    @property
    def exponents(self):
        """A dictionary mapping atomic numbers to their exponents."""
        return self._exponents

    @property
    def initials(self):
        """A dictionary mapping atomic numbers to their initial values."""
        return self._initials

    def get_nshell(self, number):
        """Get number of basis functions based on the atomic number of an atom."""
        return len(self.exponents[number])

    def get_order(self, number, ishell=None):
        """Get exponent order for an atom for its `ishell` basis function."""
        return self.orders[number] if ishell is None else self.orders[number][ishell]

    def get_initial(self, number, ishell=None):
        """Get initial value for an atom for its `ishell` basis function."""
        return (
            np.asarray(self.initials[number]) if ishell is None else self.initials[number][ishell]
        )

    def get_exponent(self, number, ishell=None):
        """Get exponent coefficient for an atom for its `ishell` basis function."""
        return (
            np.asarray(self.exponents[number]) if ishell is None else self.exponents[number][ishell]
        )

    def compute_proshell_dens(self, number, ishell, population, points, nderiv=0):
        """Compute pro-shell density on points for an atom."""
        order = self.get_order(number, ishell)
        exp = self.get_exponent(number, ishell)
        return evaluate_function(order, population, exp, points, nderiv, axis=None)

    # def compute_proatom_dens_slow(self, number, populations, points, nderiv=0, axis=0):
    #     """Compute pro-atom density on points for an atom."""
    #     orders = self.get_order(number)
    #     exps = self.get_exponent(number)
    #     return evaluate_function(orders, populations, exps, points, nderiv, axis=axis)

    def compute_proatom_dens(self, number, populations, points, nderiv=0):
        """Compute pro-atom density on points for an atom."""
        y = d = 0.0
        for i in range(self.get_nshell(number)):
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
            raise RuntimeError("The argument `nderiv` should only be 0 or 1.")

    @classmethod
    def from_function_type(cls, func_type="gauss"):
        """Construct class from basis type."""
        assert func_type in ["gauss", "slater"]
        return cls.from_yaml(DATA_PATH.joinpath(f"{func_type}.json"))

    @classmethod
    def from_file(cls, filename):
        """Construct class from a file."""
        if str(filename).endswith(".yaml"):
            extension = "yaml"
        else:
            extension = "json"
        orders, exponents, initials = load_params(filename, extension=extension)
        # check if initials values are valid.
        for number, exps in exponents.items():
            if number not in initials:
                initials[number] = np.ones_like(exps) / len(exps)
        return cls(orders, exponents, initials)

    @classmethod
    def from_yaml(cls, filename):
        """Construct from a yaml file."""
        return cls.from_file(filename)

    @classmethod
    def from_json(cls, filename):
        """Construct from a yaml file."""
        return cls.from_file(filename)
