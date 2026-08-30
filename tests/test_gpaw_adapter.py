# Copyright (C) 2026 The HORTON-Part Development Team
#
# This file is part of HORTON-Part and is distributed under GPLv3 or later.
"""Tests for the optional GPAW density adapter."""

import builtins
import os
from pathlib import Path

import numpy as np
import pytest

from horton_part.adapters.gpaw import (
    _load_gpaw,
    assemble_density_arrays,
    parse_args,
    prepare_input,
)


def test_assemble_density_arrays_with_spin():
    """Uniform and PAW blocks, including spin, must retain their order."""
    uniform = {
        "grid_points": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "grid_weights": np.array([0.5, 0.5]),
        "pseudo_density": np.array([0.8, 1.2]),
        "pseudo_spindensity": np.array([0.2, -0.1]),
        "nspins": 2,
    }
    atoms = [
        {
            "grid_points": np.array([[0.1, 0.0, 0.0]]),
            "grid_weights": np.array([0.02]),
            "density_c_cor": np.array([0.3]),
            "density_v_cor": np.array([-0.05]),
            "spindensity_v_cor": np.array([0.04]),
        },
        {
            "grid_points": np.array([[0.9, 0.0, 0.0], [1.1, 0.0, 0.0]]),
            "grid_weights": np.array([0.01, 0.01]),
            "density_c_cor": np.array([0.2, 0.1]),
            "density_v_cor": np.array([0.01, -0.02]),
            "spindensity_v_cor": np.array([-0.03, 0.02]),
        },
    ]

    result = assemble_density_arrays(uniform, atoms)

    np.testing.assert_array_equal(result["grid_sizes"], [2, 1, 2])
    np.testing.assert_allclose(result["density"], [0.8, 1.2, 0.25, 0.21, 0.08])
    np.testing.assert_allclose(result["spindensity"], [0.2, -0.1, 0.04, -0.03, 0.02])
    np.testing.assert_allclose(result["weights"], [0.5, 0.5, 0.02, 0.01, 0.01])
    assert result["points"].shape == (5, 3)


def test_parse_args_without_gpaw():
    """Constructing the CLI arguments must not import optional GPAW."""
    args = parse_args(["calculation.gpw", "density.npz"])
    assert args.fn_gpw == "calculation.gpw"
    assert args.fn_density == "density.npz"


def test_missing_gpaw_has_actionable_error(monkeypatch):
    """A GPAW-free installation should fail only when conversion is requested."""
    original_import = builtins.__import__

    def import_without_gpaw(name, *args, **kwargs):
        if name == "gpaw" or name.startswith("gpaw."):
            raise ModuleNotFoundError("No module named 'gpaw'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_gpaw)
    with pytest.raises(RuntimeError, match="intentionally not a HORTON-Part dependency"):
        _load_gpaw()


@pytest.mark.slow
def test_real_gpaw_archive_matches_reference():
    """Optionally compare conversion with an independently archived article density."""
    restart_path = os.getenv("HORTON_PART_GPAW_RESTART")
    reference_path = os.getenv("HORTON_PART_GPAW_DENSITY_REFERENCE")
    if not restart_path or not reference_path:
        pytest.skip("Set HORTON_PART_GPAW_RESTART and HORTON_PART_GPAW_DENSITY_REFERENCE")

    restart_path = Path(restart_path)
    reference_path = Path(reference_path)
    if not restart_path.is_file() or not reference_path.is_file():
        pytest.skip("Configured GPAW restart or reference density is unavailable")

    restart, _, world = _load_gpaw()
    assert world.size == 1
    atoms, calc = restart(restart_path, txt="/dev/null")
    assert calc.old
    atoms.get_potential_energy()
    actual = prepare_input(atoms, calc)

    with np.load(reference_path) as reference:
        assert set(actual) == set(reference.files)
        for key, value in actual.items():
            np.testing.assert_allclose(value, reference[key], rtol=1.0e-12, atol=1.0e-12)
