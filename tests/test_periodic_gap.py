# HORTON-PART: molecular and periodic density partitioning.
# Copyright (C) 2023-2026 The HORTON-PART Development Team
#
# This file is part of HORTON-PART and is distributed under the GNU General
# Public License, version 3 or (at your option) any later version.
"""Optional real-material parity tests against archived DensPart GaP results."""

import os
from pathlib import Path

import numpy as np
import pytest
from grid import PeriodicGrid

from horton_part.periodic import partition_periodic


GAP_ROOT = os.environ.get("HORTON_PART_GAP_ROOT")
LISA_BASIS = os.environ.get("HORTON_PART_GAP_LISA_BASIS")
SPLINE_ALL_BASIS = os.environ.get("HORTON_PART_GAP_SPLINE_ALL_BASIS")
SPLINE_BOUND_BASIS = os.environ.get("HORTON_PART_GAP_SPLINE_BOUND_BASIS")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not all((GAP_ROOT, LISA_BASIS, SPLINE_ALL_BASIS, SPLINE_BOUND_BASIS)),
        reason="GaP integration-test paths are not configured",
    ),
]


CASES = {
    "hirshfeld": ("spline_hirshfeld/gap_part.npz", "bound", 2.0e-7, 1.0e-9),
    "hirshfeld-i": ("spline_hirshfeld_i/gap_part.npz", "all", 1.0e-6, 1.0e-9),
    # Nonlinear/variational optimizer stopping points vary slightly across SciPy versions.
    "mbis": ("mbis/gap_part.npz", None, 1.0e-4, 1.0e-8),
    "lisa": ("lisa/gap_part.npz", "lisa", 1.0e-4, 1.0e-9),
    "avh-b": ("avh/gap_part.npz", "bound", 1.0e-6, 1.0e-9),
}


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


@pytest.mark.parametrize("method", CASES)
def test_gap_matches_archived_denspart_weights(method):
    root = Path(GAP_ROOT)
    reference_name, basis_kind, tolerance, inner_threshold = CASES[method]
    with np.load(root / "gap_density.npz", allow_pickle=False) as data:
        coordinates = data["atcoords"]
        numbers = data["atnums"]
        pseudo_numbers = data["atcorenums"]
        density = data["density"]
        weights = data["weights"]
        grid = PeriodicGrid(data["points"], weights, data["cellvecs"], wrap=True)
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
    )
    with np.load(root / reference_name, allow_pickle=False) as reference:
        expected = _integrated_reference_charges(reference, density, weights, pseudo_numbers)
    assert result.charges == pytest.approx(expected, abs=tolerance), (
        f"model_charges={result.model_charges!r}, iterations={result.iterations}, "
        f"last_history={result.history[-1]!r}"
    )
    assert result.charges.sum() == pytest.approx(
        pseudo_numbers.sum() - np.dot(weights, density), abs=2.0e-10
    )
    assert result.reconstruction_error < 1.0e-12
