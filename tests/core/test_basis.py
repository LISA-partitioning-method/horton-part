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
import json
import os
import tempfile

import numpy as np
import pytest

from horton_part.core.basis import BasisFuncHelper, evaluate_function, load_params


@pytest.fixture
def gauss_helper():
    return BasisFuncHelper.from_function_type("gauss")


@pytest.fixture()
def slater_helper():
    return BasisFuncHelper.from_function_type("slater")


def test_load_params():
    # Sample data to be written to the temporary JSON file
    sample_data = {
        "1": ([1, 2], [0.5, 1.5], [0.1, 0.9]),
        "2": ([2, 1], [1.0, 2.0], [0.2, 0.8]),
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmpfile:
        json.dump(sample_data, tmpfile)
        tmpfile_path = tmpfile.name  # Store the file path to use it later

    # Load parameters from the temporary file
    orders, exps, inits = load_params(tmpfile_path)

    # Asserts to check if the data loaded correctly
    assert isinstance(orders, dict)
    assert isinstance(exps, dict)
    assert isinstance(inits, dict)

    # Additional asserts can be added to compare the loaded data with `sample_data`
    os.remove(tmpfile_path)


def test_evaluate_function_with_scalar_arguments_out_size():
    n = 2
    population = 1
    alpha = 0.5
    r = np.array([0, 1, 2])
    f = evaluate_function(n, population, alpha, r)
    assert isinstance(f, np.ndarray)
    assert f.size == r.size
    assert f.ndim == 1


@pytest.mark.parametrize(
    "population, exponent, r, nderiv, expected_result",
    [
        (
            1.0,
            1.0,
            np.array([1.0]),
            0,
            np.pi ** (-1.5) * np.exp(-1),
        ),  # Test case for nderiv=0
        (
            1.0,
            1.0,
            np.array([1.0]),
            1,
            -2 * np.pi ** (-1.5) * np.exp(-1),
        ),  # Test case for nderiv=1
    ],
)
def test_evaluate_function_scalar_arguments_out_values(
    population, exponent, r, nderiv, expected_result
):
    # bs_helper = BasisFuncHelper.from_function_type("gauss")
    result = evaluate_function(2, population, exponent, r, nderiv)
    if nderiv == 1:
        assert np.isclose(result[-1], expected_result)
    else:
        assert np.isclose(result, expected_result)


@pytest.mark.parametrize(
    "populations, exponents, r, nderiv, expected_result",
    [
        (
            np.array([1.0] * 10),
            np.array([1.0] * 10),
            np.array([1.0] * 100),
            0,
            np.pi ** (-1.5) * np.exp(-1),
        ),  # Test case for nderiv=0
        (
            np.array([1.0] * 10),
            np.array([1.0] * 10),
            np.array([1.0] * 100),
            1,
            -2 * np.pi ** (-1.5) * np.exp(-1),
        ),  # Test case for nderiv=1
    ],
)
def test_evaluate_function_vector_arguments_out_values(
    populations, exponents, r, nderiv, expected_result
):
    # bs_helper = BasisFuncHelper.from_function_type("gauss")
    orders = np.ones_like(populations) * 2
    expected_result = np.ones((orders.size, r.size)) * expected_result
    result = evaluate_function(orders, populations, exponents, r, nderiv)
    if nderiv == 1:
        assert result[-1].shape == (10, 100)
        assert result[-1] == pytest.approx(expected_result)
    else:
        assert result.shape == (10, 100)
        assert result == pytest.approx(expected_result)


def test_BasisFuncHelper_from_function_type_gauss(gauss_helper):
    assert gauss_helper.get_exponent(1) == pytest.approx([5.672, 1.505, 0.5308, 0.2204])
    assert gauss_helper.get_initial(1) == pytest.approx(
        [0.04588955, 0.26113177, 0.47966863, 0.21331515]
    )
    assert gauss_helper.get_order(1) == pytest.approx(np.ones(gauss_helper.get_nshell(1)) * 2)

    assert gauss_helper.get_exponent(6) == pytest.approx(
        [148.3, 42.19, 15.33, 6.146, 0.7846, 0.2511]
    )
    assert gauss_helper.get_initial(6) == pytest.approx(
        [0.14213932, 0.58600661, 1.07985999, 0.0193642, 2.73124559, 1.44130948]
    )
    assert gauss_helper.get_exponent(7) == pytest.approx(
        [178.0, 52.42, 19.87, 1.276, 0.6291, 0.2857]
    )
    assert gauss_helper.get_initial(7) == pytest.approx(
        [0.1762005, 0.64409882, 1.00359925, 2.34863505, 1.86405143, 0.96339936]
    )
    assert gauss_helper.get_exponent(8) == pytest.approx(
        [220.1, 65.66, 25.98, 1.685, 0.6860, 0.2311]
    )
    assert gauss_helper.get_initial(8) == pytest.approx(
        [0.20051893, 0.64160829, 0.98585628, 3.01184504, 2.70306065, 0.45711097]
    )


def test_BasisFuncHelper_from_function_type_slater(slater_helper):
    assert slater_helper.get_exponent(1) == pytest.approx(
        [6.6, 4.62, 3.23, 2.26, 1.58, 1.1, 0.77, 1.0]
    )
    assert slater_helper.get_exponent(6) == pytest.approx(
        [22.8, 17.35, 13.2, 8.79, 10.83, 7.91, 5.78, 4.22, 4.51, 3.46, 2.65, 2.03, 1.56]
    )


# def test_get_gauss_nshell():
#     for number in range(1, 120):
#         if number == 1:
#             assert get_gauss_nshell(number) == 4
#         elif number == 17:
#             assert get_gauss_nshell(number) == 9
#         else:
#             assert get_gauss_nshell(number) == 8


def test_get_pro_a_k_raises_notimplementederror():
    with pytest.raises(NotImplementedError):
        evaluate_function(2, 1.0, 2.0, np.array([3.0]), 2)
