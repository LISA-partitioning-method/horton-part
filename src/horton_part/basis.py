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

__all__ = ["BasisFuncHelper"]


class BasisFuncHelper:
    def __init__(self, func_type="gauss"):
        self.func_type = func_type

    def load_exponent(self, number, nb_exp=None):
        nb_exp = nb_exp or 6
        if self.func_type == "gauss":
            return load_gauss_exponent(number, nb_exp)
        elif self.func_type == "exp":
            return load_exp_exponent(number, nb_exp)
            # return load_gauss_exponent(number, nb_exp)
        else:
            raise NotImplementedError

    def load_initial_propars(self, number, nb_exp=None):
        nb_exp = nb_exp or 6
        if self.func_type == "gauss":
            return load_gauss_initial_propars(
                number, len(self.load_exponent(number, nb_exp))
            )
        elif self.func_type == "exp":
            return load_exp_initial_propars(
                number, len(self.load_exponent(number, nb_exp))
            )
        else:
            raise NotImplementedError

    def get_nshell(self, number):
        return len(self.load_exponent(number))

    def compute_proshell_dens(self, population, exponent, points, nderiv=0):
        if self.func_type == "gauss":
            return compute_gauss_function(population, exponent, points, nderiv)
        elif self.func_type == "exp":
            return compute_exp_function(population, exponent, points, nderiv)
        else:
            raise NotImplementedError

    def compute_proatom_dens(self, populations, exponents, points, nderiv=0):
        nshell = len(populations)
        y = d = 0.0
        for k in range(nshell):
            population, exponent = populations[k], exponents[k]
            res = self.compute_proshell_dens(population, exponent, points, nderiv)
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


def load_gauss_exponent(number, nb_exp):
    """The exponents used for primitive Gaussian functions of each element."""
    assert isinstance(number, (int, np.int_)) and isinstance(nb_exp, (int, np.int_))
    param_dict = {
        1: np.array([5.672, 1.505, 0.5308, 0.2204]),
        # Li: [60.3528 14.895   5.0545  1.9759  0.0971  0.0314]
        3: np.array([60.3528, 14.895, 5.0545, 1.9759, 0.0971, 0.0314]),
        6: np.array([148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]),
        7: np.array([178.0, 52.42, 19.87, 1.276, 0.6291, 0.2857]),
        8: np.array([220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]),
        # F: [319.6233  85.5603  31.7878   2.4347   1.0167   0.3276]
        9: np.array(
            [
                319.6233,
                85.5603,
                31.7878,
                2.4347,
                1.0167,
                0.3276,
            ]
        ),
        # Si: [374.1708 118.753   69.9363   9.0231   8.3656   4.0225   0.3888   0.2045   0.0711]
        14: np.array(
            [
                4.02740000,
                0.41910000,
                8.33020000,
                88.45540000,
                0.07600000,
                213.77800000,
                0.22710000,
                592.06970000,
            ]
        ),
        # Even we use O atom data as Cl atom data, we can still obtain a reasonable result for LISA model for
        # ClO- molecule. This means it needs small exponential coefficients.
        # use O atom data for Cl
        # 17: np.array([220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]),
        # use C atom data for Cl
        # 17: np.array([148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]),
        # Cl: [439.8544 257.2415 148.3597  19.656    8.9587   2.1014   0.6936   0.5039   0.1792]
        # S: [715.0735 240.4533 120.1752  14.3796   7.0821   0.5548   0.5176   0.2499   0.1035]
        16: np.array(
            [
                715.0735,
                240.4533,
                120.1752,
                14.3796,
                7.0821,
                0.5548,
                0.5176,
                0.2499,
                0.1035,
            ]
        ),
        # Cl: [1139.4058  379.1381  151.8297   34.9379   19.5054    8.9484    0.6579    0.3952    0.1635]
        17: np.array(
            [
                1139.40580000,
                379.13810000,
                0.16350000,
                0.65790000,
                0.39520000,
                19.50540000,
                8.94840000,
                151.82970000,
            ]
        ),
        # Br: [1026.4314, 118.9876, 115.0694, 82.964, 66.2115, 64.8824, 6.4775, 5.2615, 1.6828, 0.5514, 0.2747, 0.1136,],
        35: np.array(
            [
                0.11360000,
                5.26150000,
                0.27470000,
                1026.43140000,
                1.68280000,
                0.55140000,
                64.88240000,
            ]
        ),
    }
    if number in param_dict:
        return param_dict[number]
    else:
        # use MBIS initial exponential coefficients as suggested by GISA.
        # The difference between this implentation and GISA is that Bohr radius is removed.
        # This is because atomic unit is alwasy used in cucrrent implementation.
        # The Bohr radius included in original implementation of GISA in
        # https://github.com/rbenda/ISA_multipoles is a typo.
        # (also see J. Chem. Phys. 156, 164107 (2022))
        #
        # For Li and  Cl, nb_exp = 6
        # a0 = 0.529177
        a0 = 1  # we use atomic unit
        # TODO: in Toon's paper, these coefficients are optimized by fitting Hirshfeld-I pro-atomic density
        return np.array(
            [
                2 * number ** (1 - ((i - 1) / (nb_exp - 1))) / a0
                for i in range(1, 1 + nb_exp)
            ]
        )


def load_gauss_initial_propars(number, nb_exp):
    """Create initial parameters for proatom density functions."""
    param_dict = {
        1: [
            0.04588955,
            0.26113177,
            0.47966863,
            0.21331515,
        ],
        6: [
            0.14213932,
            0.58600661,
            1.07985999,
            0.01936420,
            2.73124559,
            1.44130948,
        ],
        7: [
            0.17620050,
            0.64409882,
            1.00359925,
            2.34863505,
            1.86405143,
            0.96339936,
        ],
        8: [
            0.20051893,
            0.64160829,
            0.98585628,
            3.01184504,
            2.70306065,
            0.45711097,
        ],
        14: [
            4.66493968,
            1.43077243,
            3.02335498,
            0.92989328,
            0.29066472,
            0.57081524,
            2.85374140,
            0.23581856,
        ],
        16: [
            0.29455963,
            0.58698056,
            0.80997645,
            1.57119984,
            5.92511019,
            2.30623427,
            2.25300903,
            2.07579989,
            0.17713248,
        ],
        17: [
            0.16005908,
            0.49615207,
            0.67085328,
            4.54949229,
            2.68454233,
            0.93361892,
            6.49142386,
            1.01385860,
        ],
        35: [
            0.26002704,
            17.04051595,
            3.02612546,
            1.54627807,
            1.25096851,
            5.26425613,
            6.61182792,
        ],
    }
    if number in param_dict:
        return param_dict[number]
    else:
        # from https://github.com/rbenda/ISA_multipoles.
        return np.ones(nb_exp, float) * number / nb_exp


def compute_gauss_function(population, exponent, points, nderiv=0):
    """The primitive Gaussian function with exponent of `alpha_k`."""
    f = population * (exponent / np.pi) ** 1.5 * np.exp(-exponent * points**2)
    if nderiv == 0:
        return f
    elif nderiv == 1:
        return f, -2 * exponent * points * f
    else:
        raise NotImplementedError


def load_exp_exponent(number, nb_exp):
    """The exponents used for primitive Gaussian functions of each element."""
    assert isinstance(number, (int, np.int_)) and isinstance(nb_exp, (int, np.int_))
    param_dict = {
        # TZP fc: Triple zeta forzen core 1 polarization function
        # See https://www.scm.com/zorabasis/periodic.tzpfc.html
        1: np.array([3.16, 2.09, 1.38, 1.50]),
        6: np.array([10.80, 11.59, 7.59, 4.97, 4.79, 3.35, 2.34, 1.64]),
        7: np.array([12.76, 13.77, 9.06, 5.97, 5.77, 4.05, 2.85, 2.00]),
        8: np.array([14.72, 15.80, 10.35, 6.78, 6.54, 4.57, 3.20, 2.24]),
        9: np.array([16.66, 16.24, 9.81, 5.93, 5.30, 3.46, 2.26, 1.48]),
        16: np.array(
            [26.90, 26.93, 16.64, 15.19, 10.12, 8.95, 6.26, 4.37, 3.82, 2.76, 2.00]
        ),
    }
    if number in param_dict:
        return param_dict[number]
    else:
        return np.array(
            [2 * number ** (1 - ((i - 1) / (nb_exp - 1))) for i in range(1, 1 + nb_exp)]
        )


def load_exp_initial_propars(number, nb_exp):
    """Create initial parameters for proatom density functions."""
    param_dict = {}
    if number in param_dict:
        return param_dict[number]
    else:
        return np.ones(nb_exp, float) * number / nb_exp


def compute_exp_function(population, exponent, points, nderiv=0):
    """The primitive Gaussian function with exponent of `alpha_k`."""
    f = population * (exponent**3 / 8 / np.pi) * np.exp(-exponent * points)
    if nderiv == 0:
        return f
    elif nderiv == 1:
        return f, -exponent * f
    else:
        raise NotImplementedError
