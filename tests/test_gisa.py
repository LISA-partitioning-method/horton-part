#!/usr/bin/env python
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
import numpy as np
import pytest
from horton_part.basis import BasisFuncHelper


def test_alpha():
    bs_helper = BasisFuncHelper()
    assert bs_helper.load_exponent(1) == pytest.approx([5.672, 1.505, 0.5308, 0.2204])
    assert bs_helper.load_exponent(6) == pytest.approx(
        [148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]
    )
    assert bs_helper.load_exponent(7) == pytest.approx(
        [178.0, 52.42, 19.87, 1.276, 0.6291, 0.2857]
    )
    assert bs_helper.load_exponent(8) == pytest.approx(
        [220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]
    )

    with pytest.raises(AssertionError):
        bs_helper.load_exponent(1.1)

    with pytest.raises(AssertionError):
        bs_helper.load_exponent(2, 1.1)


# def test_get_gauss_nshell():
#     for number in range(1, 120):
#         if number == 1:
#             assert get_gauss_nshell(number) == 4
#         elif number == 17:
#             assert get_gauss_nshell(number) == 9
#         else:
#             assert get_gauss_nshell(number) == 8


@pytest.mark.parametrize(
    "D_k, alpha_k, r, nderiv, expected_result",
    [
        (1.0, 1.0, 1.0, 0, np.pi ** (-1.5) * np.exp(-1)),  # Test case for nderiv=0
        (1.0, 1.0, 1.0, 1, -2 * np.pi ** (-1.5) * np.exp(-1)),  # Test case for nderiv=1
    ],
)
def test_get_pro_a_k(D_k, alpha_k, r, nderiv, expected_result):
    bs_helper = BasisFuncHelper()
    result = bs_helper.compute_proshell_dens(D_k, alpha_k, r, nderiv)
    if nderiv == 1:
        assert np.isclose(result[-1], expected_result)
    else:
        assert np.isclose(result, expected_result)


def test_get_pro_a_k_raises_notimplementederror():
    bs_helper = BasisFuncHelper()
    with pytest.raises(NotImplementedError):
        bs_helper.compute_proshell_dens(1.0, 2.0, 3.0, 2)
