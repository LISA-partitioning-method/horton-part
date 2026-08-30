# HORTON-PART: molecular and periodic density partitioning.
# Copyright (C) 2023-2026 The HORTON-PART Development Team
#
# This file is part of HORTON-PART and is distributed under the GNU General
# Public License, version 3 or (at your option) any later version.
"""Optional real-material parity tests against archived DensPart KBBF results."""

import os
from pathlib import Path

import numpy as np
import pytest
from grid import PeriodicGrid

from horton_part.periodic import partition_periodic


KBBF_ROOT = os.environ.get("HORTON_PART_KBBF_ROOT")
MBIS_REFERENCE = os.environ.get("HORTON_PART_KBBF_MBIS_REFERENCE")
LISA_BASIS = os.environ.get("HORTON_PART_KBBF_LISA_BASIS")
SPLINE_ALL_BASIS = os.environ.get("HORTON_PART_KBBF_SPLINE_ALL_BASIS")
SPLINE_BOUND_BASIS = os.environ.get("HORTON_PART_KBBF_SPLINE_BOUND_BASIS")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not all(
            (
                KBBF_ROOT,
                MBIS_REFERENCE,
                LISA_BASIS,
                SPLINE_ALL_BASIS,
                SPLINE_BOUND_BASIS,
            )
        ),
        reason="KBBF integration-test paths are not configured",
    ),
]


CASES = {
    "hirshfeld": ("spline_hirshfeld/kbbf_part.npz", "bound", 2.0e-7, 1.0e-9),
    "hirshfeld-i": ("spline_hirshfeld_i/kbbf_part.npz", "all", 1.0e-6, 1.0e-9),
    # Nonlinear/variational optimizer stopping points vary slightly across SciPy versions.
    "mbis": (None, None, 1.0e-4, 1.0e-8),
    # The 145-parameter KBBF LISA fit is exceptionally flat: independently converged
    # SciPy endpoints differ by 3.11e-4 electrons but only 1.27e-7 in the objective.
    "lisa": ("lisa/kbbf_part.npz", "lisa", 5.0e-4, 1.0e-9),
    "avh-b": ("avh/kbbf_part.npz", "bound", 1.0e-6, 1.0e-9),
}


@pytest.fixture(scope="module")
def kbbf_density():
    with np.load(Path(KBBF_ROOT) / "kbbf_density.npz", allow_pickle=False) as data:
        coordinates = data["atcoords"]
        numbers = data["atnums"]
        pseudo_numbers = data["atcorenums"]
        density = data["density"]
        weights = data["weights"]
        grid = PeriodicGrid(data["points"], weights, data["cellvecs"], wrap=True)
    return coordinates, numbers, pseudo_numbers, density, weights, grid


def _integrated_reference_charges(reference, density, weights, pseudo_numbers):
    aim_weights = np.asarray(reference["aim_weights"])
    weight_sums = aim_weights.sum(axis=0)
    normalized = np.divide(
        aim_weights,
        weight_sums,
        out=np.zeros_like(aim_weights),
        where=weight_sums > 0.0,
    )
    populations = np.einsum("ap,p,p->a", normalized, density, weights)
    return pseudo_numbers - populations


@pytest.mark.parametrize(
    "return_weights", [False, True], ids=["local-only", "dense-weights"]
)
@pytest.mark.parametrize("method", CASES)
def test_kbbf_matches_archived_denspart_weights(method, return_weights, kbbf_density):
    coordinates, numbers, pseudo_numbers, density, weights, grid = kbbf_density
    reference_name, basis_kind, tolerance, inner_threshold = CASES[method]
    basis = {
        None: None,
        "lisa": LISA_BASIS,
        "all": SPLINE_ALL_BASIS,
        "bound": SPLINE_BOUND_BASIS,
    }[basis_kind]
    result = partition_periodic(
        method,
        coordinates,
        numbers,
        grid,
        density,
        basis=basis,
        pseudo_numbers=pseudo_numbers,
        threshold=1.0e-8,
        inner_threshold=inner_threshold,
        maxiter=2000,
        inner_maxiter=2000,
        return_weights=return_weights,
    )
    reference_path = (
        Path(MBIS_REFERENCE) if method == "mbis" else Path(KBBF_ROOT) / reference_name
    )
    with np.load(reference_path, allow_pickle=False) as reference:
        expected = _integrated_reference_charges(
            reference, density, weights, pseudo_numbers
        )
    assert result.charges == pytest.approx(expected, abs=tolerance), (
        f"model_charges={result.model_charges!r}, iterations={result.iterations}, "
        f"last_history={result.history[-1]!r}"
    )
    assert result.charges.sum() == pytest.approx(
        pseudo_numbers.sum() - np.dot(weights, density), abs=2.0e-10
    )
    if return_weights:
        assert result.aim_weights is not None
        assert result.reconstruction_error < 1.0e-12
    else:
        assert result.aim_weights is None
        assert np.isnan(result.reconstruction_error)
