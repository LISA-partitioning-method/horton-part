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
"""Utility Functions"""

import warnings

import numpy as np
import yaml
from importlib_resources import files

__all__ = [
    "typecheck_geo",
    "radius_becke",
    "radius_covalent",
    "wpart_schemes",
    "compute_quantities",
    "check_pro_atom_parameters",
    "check_for_pro_error",
    "check_for_grad_error",
    "check_for_hessian_error",
    "NEGATIVE_CUTOFF",
    "POPULATION_CUTOFF",
    "ANGSTROM",
    "PERIODIC_TABLE",
    "NegativeDensityError",
    "NotMonotonicDecayError",
    "fix_propars",
]


# Constants
NEGATIVE_CUTOFF = -1e-12
POPULATION_CUTOFF = 1e-4
ANGSTROM = 1.0e-10 / 0.5291772083e-10
JSON_DATA_PATH = files("horton_part.data")
DATA_PATH = files("horton_part.data")

with open(DATA_PATH.joinpath("config.yaml")) as f:
    docu = yaml.safe_load(f)
PERIODIC_TABLE = docu["PERIODIC_TABLE"]


def wpart_schemes(scheme):
    if scheme == "h":
        from .hirshfeld import HirshfeldWPart

        wpart = HirshfeldWPart
    elif scheme == "hi":
        from .hirshfeld_i import HirshfeldIWPart

        wpart = HirshfeldIWPart
    elif scheme == "is":
        from .isa import ISAWPart

        wpart = ISAWPart
    elif scheme == "mbis":
        from .mbis import MBISWPart

        wpart = MBISWPart
    elif scheme == "nlis":
        from .nlis import NLISWPart

        wpart = NLISWPart
    elif scheme == "gmbis":
        from .gmbis import GMBISWPart

        wpart = GMBISWPart
    elif scheme == "b":
        from .becke import BeckeWPart

        wpart = BeckeWPart
    elif scheme == "lisa":
        from .lisa import LinearISAWPart

        wpart = LinearISAWPart
    elif scheme == "glisa":
        from .lisa_g import GlobalLinearISAWPart

        wpart = GlobalLinearISAWPart
    elif scheme == "gisa":
        from .gisa import GaussianISAWPart

        wpart = GaussianISAWPart
    return wpart


def typecheck_geo(
    coordinates=None,
    numbers=None,
    pseudo_numbers=None,
    need_coordinates=True,
    need_numbers=True,
    need_pseudo_numbers=True,
):
    """Type check a molecular geometry specification

    **Arguments:**

    coordinates
         A (N, 3) float array with Cartesian coordinates of the atoms.

    numbers
         A (N,) int vector with the atomic numbers.

    **Optional arguments:**

    pseudo_numbers
         A (N,) float array with pseudo-potential core charges.

    need_coordinates
         When set to False, the coordinates can be None, are not type checked
         and not returned.

    need_numbers
         When set to False, the numbers can be None, are not type checked
         and not returned.

    need_pseudo_numbers
         When set to False, the pseudo_numbers can be None, are not type
         checked and not returned.

    **Returns:** ``[natom]`` + all arguments that were type checked. The
    pseudo_numbers argument is converted to a floating point array.
    """
    # Determine natom
    if coordinates is not None:
        natom = len(coordinates)
    elif numbers is not None:
        natom = len(numbers)
    elif pseudo_numbers is not None:
        natom = len(pseudo_numbers)
    else:
        raise TypeError("At least one argument is required and should not be None")

    # Typecheck coordinates:
    if coordinates is None:
        if need_coordinates:
            raise TypeError("Coordinates can not be None.")
    else:
        if coordinates.shape != (natom, 3) or not issubclass(coordinates.dtype.type, float):
            raise TypeError("The argument centers must be a float array with shape (natom,3).")

    # Typecheck numbers
    if numbers is None:
        if need_numbers:
            raise TypeError("Numbers can not be None.")
    else:
        if numbers.shape != (natom,) or not issubclass(numbers.dtype.type, np.int64):
            raise TypeError("The argument numbers must be a vector with length natom.")

    # Typecheck pseudo_numbers
    if pseudo_numbers is None:
        if need_pseudo_numbers:
            pseudo_numbers = numbers.astype(float)
    else:
        if pseudo_numbers.shape != (natom,):
            raise TypeError("The argument pseudo_numbers must be a vector with length natom.")
        if not issubclass(pseudo_numbers.dtype.type, float):
            pseudo_numbers = pseudo_numbers.astype(float)

    # Collect return values
    result = [
        natom,
    ]
    if need_coordinates:
        result.append(coordinates)
    if need_numbers:
        result.append(numbers)
    if need_pseudo_numbers:
        result.append(pseudo_numbers)
    return result


# cov_radius_slater if present, else cov_radius_cordero if present, else none
radius_becke = {
    1: 0.25,
    2: 0.28,
    3: 1.45,
    4: 1.05,
    5: 0.85,
    6: 0.7,
    7: 0.65,
    8: 0.6,
    9: 0.5,
    10: 0.58,
    11: 1.8,
    12: 1.5,
    13: 1.25,
    14: 1.1,
    15: 1.0,
    16: 1.0,
    17: 1.0,
    18: 1.06,
    19: 2.2,
    20: 1.8,
    21: 1.6,
    22: 1.4,
    23: 1.35,
    24: 1.4,
    25: 1.4,
    26: 1.4,
    27: 1.35,
    28: 1.35,
    29: 1.35,
    30: 1.35,
    31: 1.3,
    32: 1.25,
    33: 1.15,
    34: 1.15,
    35: 1.15,
    36: 1.16,
    37: 2.35,
    38: 2.0,
    39: 1.8,
    40: 1.55,
    41: 1.45,
    42: 1.45,
    43: 1.35,
    44: 1.3,
    45: 1.35,
    46: 1.4,
    47: 1.6,
    48: 1.55,
    49: 1.55,
    50: 1.45,
    51: 1.45,
    52: 1.4,
    53: 1.4,
    54: 1.4,
    55: 2.6,
    56: 2.15,
    57: 1.95,
    58: 1.85,
    59: 1.85,
    60: 1.85,
    61: 1.85,
    62: 1.85,
    63: 1.85,
    64: 1.8,
    65: 1.75,
    66: 1.75,
    67: 1.75,
    68: 1.75,
    69: 1.75,
    70: 1.75,
    71: 1.75,
    72: 1.55,
    73: 1.45,
    74: 1.35,
    75: 1.35,
    76: 1.3,
    77: 1.35,
    78: 1.35,
    79: 1.35,
    80: 1.5,
    81: 1.9,
    82: 1.46,
    83: 1.6,
    84: 1.9,
    85: 1.5,
    86: 1.5,
    87: 2.6,
    88: 2.15,
    89: 1.95,
    90: 1.8,
    91: 1.8,
    92: 1.75,
    93: 1.75,
    94: 1.75,
    95: 1.75,
    96: 1.69,
    97: None,
    98: None,
    99: None,
    100: None,
    101: None,
    102: None,
    103: None,
    104: None,
    105: None,
    106: None,
    107: None,
    108: None,
    109: None,
    110: None,
    111: None,
    112: None,
    113: None,
    114: None,
    115: None,
    116: None,
    117: None,
    118: None,
}
radius_becke = {k: v if v is None else v * ANGSTROM for k, v in list(radius_becke.items())}


# cov_radius_cordero
radius_covalent = {
    1: 0.31,
    2: 0.28,
    3: 1.28,
    4: 0.96,
    5: 0.84,
    6: 0.74453374,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    10: 0.58,
    11: 1.66,
    12: 1.41,
    13: 1.21,
    14: 1.11,
    15: 1.07,
    16: 1.05,
    17: 1.02,
    18: 1.06,
    19: 2.03,
    20: 1.76,
    21: 1.7,
    22: 1.6,
    23: 1.53,
    24: 1.39,
    25: 1.553504,
    26: 1.4841791,
    27: 1.29754491,
    28: 1.24,
    29: 1.32,
    30: 1.22,
    31: 1.22,
    32: 1.2,
    33: 1.19,
    34: 1.2,
    35: 1.2,
    36: 1.16,
    37: 2.2,
    38: 1.95,
    39: 1.9,
    40: 1.75,
    41: 1.64,
    42: 1.54,
    43: 1.47,
    44: 1.46,
    45: 1.42,
    46: 1.39,
    47: 1.45,
    48: 1.44,
    49: 1.42,
    50: 1.39,
    51: 1.39,
    52: 1.38,
    53: 1.39,
    54: 1.4,
    55: 2.44,
    56: 2.15,
    57: 2.07,
    58: 2.04,
    59: 2.03,
    60: 2.01,
    61: 1.99,
    62: 1.98,
    63: 1.98,
    64: 1.96,
    65: 1.94,
    66: 1.92,
    67: 1.92,
    68: 1.89,
    69: 1.9,
    70: 1.87,
    71: 1.87,
    72: 1.75,
    73: 1.7,
    74: 1.62,
    75: 1.51,
    76: 1.44,
    77: 1.41,
    78: 1.36,
    79: 1.36,
    80: 1.32,
    81: 1.45,
    82: 1.46,
    83: 1.48,
    84: 1.4,
    85: 1.5,
    86: 1.5,
    87: 2.6,
    88: 2.21,
    89: 2.15,
    90: 2.06,
    91: 2.0,
    92: 1.96,
    93: 1.9,
    94: 1.87,
    95: 1.8,
    96: 1.69,
    97: None,
    98: None,
    99: None,
    100: None,
    101: None,
    102: None,
    103: None,
    104: None,
    105: None,
    106: None,
    107: None,
    108: None,
    109: None,
    110: None,
    111: None,
    112: None,
    113: None,
    114: None,
    115: None,
    116: None,
    117: None,
    118: None,
}
radius_covalent = {k: v if v is None else v * ANGSTROM for k, v in list(radius_covalent.items())}


def compute_quantities(
    density,
    pro_atom_params,
    basis_functions,
    density_cutoff,
    do_sick=True,
    do_ratio=True,
    do_ln_ratio=True,
):
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

    sick = ratio = ln_ratio = None
    if do_sick:
        sick = (density < density_cutoff) | (pro_density < density_cutoff)
    with np.errstate(all="ignore"):
        if do_ratio:
            ratio = np.divide(density, pro_density, out=np.zeros_like(density), where=~sick)
        if do_ln_ratio:
            ln_ratio = np.log(ratio, out=np.zeros_like(density), where=~sick)
    return pro_shells, pro_density, sick, ratio, ln_ratio


def check_pro_atom_parameters(
    pro_atom_params,
    basis_functions=None,
    total_population=None,
    pro_atom_density=None,
    check_monotonicity=True,
    check_dens_negativity=True,
    check_propars_negativity=True,
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
    check_dens_negativity: bool, optional
        Check if the density is non-negativity.
    check_propars_negativity: bool, optional
        Check if the propars is negative.

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
    if check_propars_negativity and (pro_atom_params < NEGATIVE_CUTOFF).any():
        warnings.warn("Not all pro-atom parameters are positive!")

    # Calculate pro-atom density if not provided
    if basis_functions is not None and pro_atom_density is None:
        if pro_atom_params.size != basis_functions.shape[0]:
            raise ValueError(
                "Length of pro_atom_params does not match the number of basis functions"
            )
        pro_atom_density = (basis_functions * pro_atom_params[:, None]).sum(axis=0)

    # Check for negative pro-atom density
    if (
        check_dens_negativity
        and pro_atom_density is not None
        and (pro_atom_density < NEGATIVE_CUTOFF).any()
    ):
        raise RuntimeError("Negative pro-atom density found!")

    # Check if the sum of pro-atom parameters matches total population
    if total_population is not None and not np.allclose(
        np.sum(pro_atom_params), total_population, atol=POPULATION_CUTOFF
    ):
        warnings.warn(
            r"The sum of pro-atom parameters is not equal to atomic population."
            rf"The difference is {np.sum(pro_atom_params) - total_population}"
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


class NegativeDensityError(RuntimeError):
    """Negative density error."""

    pass


class NotMonotonicDecayError(RuntimeError):
    """The pro-density is not monotonic decay error."""

    pass


# Function to update propars
def fix_propars(exp_array, propars, delta):
    sorted_indices = np.argsort(exp_array)
    check_diffused = True
    fixed_indices = []
    for index in sorted_indices:
        # Check if the corresponding c_ak is non-zero and ensure positivity
        if check_diffused and np.abs(propars[index]) < 1e-4 and delta[index] < 0.0:
            fixed_indices.append(index)
        else:
            check_diffused = False
            # if check_diffused:
            #     assert propars[index] > 0.0
    return fixed_indices
