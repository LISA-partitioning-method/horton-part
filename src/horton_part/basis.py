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
import warnings
import json
from importlib_resources import files

__all__ = [
    "BasisFuncHelper",
    "compute_quantities",
    "check_pro_atom_parameters",
    "check_for_pro_error",
    "check_for_grad_error",
    "check_for_hessian_error",
    "NEGATIVE_CUTOFF",
    "POPULATION_CUTOFF",
]

NEGATIVE_CUTOFF = -1e-12
POPULATION_CUTOFF = 1e-4

# Constants
DEFAULT_NB_EXP = 6
JSON_DATA_PATH = files("horton_part.data")
FILE_PATHS = {
    "gauss": {
        "exponent": JSON_DATA_PATH.joinpath("gauss_exponents.json"),
        "initial": JSON_DATA_PATH.joinpath("gauss_initials.json"),
    },
    "slater": {
        # https://www.scm.com/zorabasis/periodic.qz4pae.html
        "exponent": JSON_DATA_PATH.joinpath("slater_exponents.json"),
        "initial": JSON_DATA_PATH.joinpath("slater_initials.json"),
    },
}


class BasisFuncHelper:
    """Helper class for basis function."""

    def __init__(self, func_type="gauss", exponents=None, initials=None):
        self.func_type = func_type
        self.exp_order = {"gauss": 2, "slater": 1}[func_type]
        self._cache = {}

        if exponents is None:
            exponents = {
                iatom: self.load_exponent(iatom)
                for iatom in [1, 3, 6, 7, 8, 9, 14, 16, 17, 35]
            }
        self.exponents = exponents

        orders = {
            iatom: np.ones_like(exps) / len(exps)
            for iatom, exps in self.exponents.items()
        }
        self.orders = orders

        if initials is None:
            initials = {
                iatom: self.load_initials(iatom) for iatom in self.exponents.keys()
            }
        self.initials = initials

        print(self.exponents)

    def load_initials(self, number):
        """Load initial populations of exponential functions for an atom."""
        exps = self.exponents[number]
        key = ("pops", number)
        if key not in self._cache:
            self._cache[key] = self._load_params(number, len(exps), "initial")
            assert len(self._cache[key]) == len(exps)
        return self._cache[key]

    def load_exponent(self, number, nb_exp=None):
        key = ("exps", number)
        if key not in self._cache:
            self._cache[key] = self._load_params(number, nb_exp)
        return self._cache[key]

    def _load_params(self, number, nb_exp=None, param_type="exponent"):
        nb_exp = nb_exp or DEFAULT_NB_EXP
        json_file = FILE_PATHS[self.func_type][param_type]

        with open(json_file) as file:
            param_dict = json.load(file)
        param_dict = {int(k): np.asarray(v) for k, v in param_dict.items()}

        if number in param_dict:
            return param_dict[number]
        else:
            return self._generate_default_params(number, nb_exp, param_type)

    @staticmethod
    def _generate_default_params(number, nb_exp, param_type):
        # Fallback logic for default parameters
        if param_type == "exponent":
            return np.array(
                [
                    2 * number ** (1 - (i - 1) / (nb_exp - 1))
                    for i in range(1, nb_exp + 1)
                ]
            )
        elif param_type == "initial":
            return np.ones(nb_exp, float) * number / nb_exp
        else:
            raise ValueError(f"Unsupported param_type: {param_type}")

    def get_nshell(self, number):
        """Get number of basis functions based on the atomic number of an atom."""
        return len(self.exponents[number])

    def compute_proshell_dens(self, number, ishell, population, points, nderiv=0):
        """Compute pro-shell density on points for an atom."""
        return self.evaluate_general_function(
            self.exp_order, population, self.exponents[number][ishell], points, nderiv
        )

    def compute_proatom_dens(self, number, populations, points, nderiv=0):
        """Compute pro-atom density on points for an atom."""
        nshell = len(populations)
        y = d = 0.0
        for k in range(nshell):
            res = self.compute_proshell_dens(number, k, populations[k], points, nderiv)
            if nderiv == 0:
                y += res
            elif nderiv == 1:
                y += res[0]
                d += res[1]
            else:
                raise NotImplementedError
        return {0: y, 1: (y, d)}[nderiv]

    @staticmethod
    def evaluate_general_function(n, population, alpha, r, nderiv=0):
        r"""Evaluate general function and its derivative.

        .. math::

            f(\mathbf{r}) = N(\alpha, n) \exp^{-\alpha |\mathbf{r}|^n}

            \int N(\alpha, n)  f(\mathbf{r}) d\mathbf{r} =  4 \pi * \int_0^{\infty} r^2  \exp^{-\alpha r^n} = 1

            N(\alpha, n) = \frac{n \alpha^{3/n}}{4 \pi \Gamma(3/n)}

            N(\alpha, 1) = \frac{\alpha^3}{8 \pi}

            N(\alpha, 2) = (\frac{\alpha}{\pi})^{3/2}

            \frac{\partial{f(\mathbf{r})}}{\partial{r}} = - n r^{n-1} \alpha f(\mathbf{r})


        Parameters
        ----------
        population: float
            Constant representing population.
        n: float
            Order, positive
        alpha: float
            Exponential coefficient
        r:
            Points
        nderiv: int
            The order of derivative

        """
        prefactor = n * alpha ** (3 / n) / (4 * np.pi * gamma(3 / n))
        f = population * prefactor * np.exp(-alpha * r**n)
        df = -n * r ** (n - 1) * f
        return {0: f, 1: (f, df)}[nderiv]


def compute_quantities(density, pro_atom_params, basis_functions, density_cutoff):
    """
    Compute various quantities based on input density, pro-atom parameters, and basis functions.

    Parameters
    ----------
    density : ndarray
        The density array.
    pro_atom_params : ndarray
        Array of pro-atom parameters.
    basis_functions : ndarray
        Array of basis functions.
    density_cutoff : float, optional
        The cutoff value for density and pro-atom density, below which values are considered negligible.

    Returns
    -------
    pro_shells : ndarray
        Calculated pro-atom shell values.
    pro_density : ndarray
        Calculated pro-atom density.
    sick : ndarray
        Boolean array indicating where density or pro-density is below the cutoff.
    ratio : ndarray
        Ratio of density to pro-atom density, with adjustments for sick values.
    ln_ratio : ndarray
        Natural logarithm of the ratio, with adjustments for sick values.
    """
    pro_atom_params = np.asarray(pro_atom_params).flatten()
    pro_shells = basis_functions * pro_atom_params[:, None]
    pro_density = np.einsum("ij->j", pro_shells)
    sick = (density < density_cutoff) | (pro_density < density_cutoff)
    with np.errstate(all="ignore"):
        ratio = np.divide(density, pro_density, out=np.zeros_like(density), where=~sick)
        ln_ratio = np.log(ratio, out=np.zeros_like(density), where=~sick)
    return pro_shells, pro_density, sick, ratio, ln_ratio


def check_pro_atom_parameters(
    pro_atom_params,
    basis_functions=None,
    total_population=None,
    pro_atom_density=None,
    check_monotonicity=True,
):
    """
    Check the validity of pro-atom parameters.

    This function warns if any pro-atom parameters are negative and raises a
    RuntimeError if the pro-atom density contains negative values. It also warns
    if the sum of pro-atom parameters does not match the expected total atomic
    population.

    Parameters
    ----------
    pro_atom_params : ndarray
        Array of pro-atom parameters.
    basis_functions : ndarray, optional
        Basis functions on the grid.
    total_population : float, optional
        Total atomic population to compare against the sum of pro-atom parameters.
    pro_atom_density : ndarray, optional
        Pre-calculated pro-atom density array. If not provided, it is calculated
        from pro_atom_params and basis_functions.
    check_monotonicity: bool, optional
        Check if the density is monotonically decreased w.r.t. radial radius.

    Raises
    ------
    RuntimeError
        If negative values are found in the pro-atom density.

    Warnings
    --------
    UserWarning
        If any pro-atom parameters are negative.
        If the sum of pro-atom parameters does not match the total_population.

    Returns
    -------
    None
    """
    # Validate inputs
    pro_atom_params = np.asarray(pro_atom_params)
    if pro_atom_params.ndim != 1:
        raise ValueError("pro_atom_params must be a 1D array")

    if basis_functions is not None:
        basis_functions = np.asarray(basis_functions)
        if basis_functions.ndim != 2:
            raise ValueError("basis_functions must be a 2D array")

    if pro_atom_density is not None:
        pro_atom_density = np.asarray(pro_atom_density)
        if pro_atom_density.ndim != 1:
            raise ValueError("pro_atom_density must be a 1D array")

    # Check if pro-atom parameters are positive
    if (pro_atom_params < NEGATIVE_CUTOFF).any():
        warnings.warn("Not all pro-atom parameters are positive!")

    # Calculate pro-atom density if not provided
    if basis_functions is not None and pro_atom_density is None:
        if pro_atom_params.size != basis_functions.shape[0]:
            raise ValueError(
                "Length of pro_atom_params does not match the number of basis functions"
            )
        pro_atom_density = (basis_functions * pro_atom_params[:, None]).sum(axis=0)

    # Check for negative pro-atom density
    if pro_atom_density is not None and (pro_atom_density < NEGATIVE_CUTOFF).any():
        raise RuntimeError("Negative pro-atom density found!")

    # Check if the sum of pro-atom parameters matches total population
    if total_population is not None and not np.allclose(
        np.sum(pro_atom_params), total_population, atol=POPULATION_CUTOFF
    ):
        warnings.warn(
            "The sum of pro-atom parameters is not equal to atomic population"
        )

    if check_monotonicity and pro_atom_density is not None:
        if (pro_atom_density[:-1] - pro_atom_density[1:] < NEGATIVE_CUTOFF).any():
            raise RuntimeError("Pro-atom density should be monotonically decreasing.")


def check_for_pro_error(pro, as_warn=True):
    """Check for non-monotonic and non-negative density"""
    if (pro < NEGATIVE_CUTOFF).any():
        if as_warn:
            warnings.warn("Negative pro-atom density found during optimization!")
        else:
            raise RuntimeError("Negative pro-atom density found!")
    if (pro[:-1] - pro[1:] < NEGATIVE_CUTOFF).any():
        if as_warn:
            warnings.warn("Pro-atom density should be monotonically decreasing.")
        else:
            raise RuntimeError("Pro-atom density should be monotonically decreasing.")


def check_for_grad_error(grad):
    """Check for non-monotonic density and negative gradient errors."""
    if (grad < NEGATIVE_CUTOFF).any():
        raise RuntimeError("Negative gradient detected.")


def check_for_hessian_error(hess):
    """Check for negative-eigenvalue hessian errors."""
    if (np.linalg.eigvals(hess) <= NEGATIVE_CUTOFF).any():
        # raise RuntimeError("All eigenvalues of Hessian matrix are not all")
        warnings.warn("All eigenvalues of Hessian matrix are not all")
