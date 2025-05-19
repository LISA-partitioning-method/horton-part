# HORTON-PART: molecular density partition schemes based on HORTON package.
# Copyright (C) 2023-2025 The HORTON-PART Development Team
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

from horton_part.core.basis import ExpBasisFuncHelper, evaluate_function, load_params
from horton_part.utils import DATA_PATH


@pytest.mark.parametrize("type", ["gauss", "slater"])
def test_basis_files(type):
    # Load the data based on the format
    d1 = load_params(DATA_PATH / f"{type}.json", "json")
    d2 = load_params(DATA_PATH / f"{type}.yaml", "yaml")

    for info1, info2 in zip(d1, d2):
        assert len(info1) == len(info2)
        for k, v in info1.items():
            assert k in info2
            assert v == pytest.approx(info2[k])


@pytest.fixture
def gauss_helper():
    return ExpBasisFuncHelper.from_function_type("gauss")


@pytest.fixture()
def slater_helper():
    return ExpBasisFuncHelper.from_function_type("slater")


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


def test_load_params_yaml():
    # Sample data to be written to the temporary JSON file
    sample_data = {
        "1": ([1, 2], [0.5, 1.5], [0.1, 0.9]),
        "2": ([2, 1], [1.0, 2.0], [0.2, 0.8]),
    }

    # Create a temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".yaml") as tmpfile:
        json.dump(sample_data, tmpfile)
        tmpfile_path = tmpfile.name  # Store the file path to use it later

    # Load parameters from the temporary file
    orders, exps, inits = load_params(tmpfile_path, extension="yaml")

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


def test_BasisFuncHelper_from_function_type_slater(slater_helper):
    assert slater_helper.get_exponent(1) == pytest.approx(
        [6.6, 4.62, 3.23, 2.26, 1.58, 1.1, 0.77, 1.0]
    )
    assert slater_helper.get_exponent(6) == pytest.approx(
        [22.8, 17.35, 13.2, 8.79, 10.83, 7.91, 5.78, 4.22, 4.51, 3.46, 2.65, 2.03, 1.56]
    )


def test_get_pro_a_k_raises_notimplementederror():
    with pytest.raises(NotImplementedError):
        evaluate_function(2, 1.0, 2.0, np.array([3.0]), 2)
