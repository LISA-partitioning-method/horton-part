# HORTON-PART: molecular and periodic density partitioning.
# Copyright (C) 2023-2026 The HORTON-PART Development Team
#
# This file is part of HORTON-PART and is distributed under the GNU General
# Public License, version 3 or (at your option) any later version.
"""Tests for the common periodic stockholder engine."""

import numpy as np
import pytest
from grid import PeriodicGrid

from horton_part.periodic import load_lisa_basis, load_spline_proatoms, partition_periodic
from horton_part.periodic.basis import ExponentialShape
from horton_part.scripts.partition_periodic import main as periodic_main


def _uniform_periodic_grid(length=8.0, npoint=20):
    axis = (np.arange(npoint) + 0.5) * length / npoint
    points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    weights = np.full(len(points), length**3 / len(points))
    return PeriodicGrid(points, weights, np.eye(3) * length, wrap=True)


def _periodic_density(grid, center, shape, cutoff=1.0e-10):
    localgrid = grid.get_localgrid(center, shape.cutoff(cutoff, 1.0))
    radii = np.linalg.norm(localgrid.points - center, axis=1)
    density = np.zeros(len(grid.points))
    np.add.at(density, localgrid.indices, shape.evaluate(radii))
    return density


def _spline_basis():
    radii = np.linspace(1.0e-4, 12.0, 801)
    step = radii[1] - radii[0]
    radial_weights = np.full_like(radii, step)
    radial_weights[[0, -1]] *= 0.5
    volume_weights = 4.0 * np.pi * radii**2 * radial_weights
    states = []
    for charge, exponent in ((-1, 0.4), (0, 1.0), (1, None)):
        electrons = 1 - charge
        if electrons:
            density = np.exp(-exponent * radii**2)
            density *= electrons / np.dot(volume_weights, density)
        else:
            density = np.zeros_like(radii)
        states.append(
            {
                "charge": charge,
                "electrons": electrons,
                "density": density.tolist(),
                "bound_to_electron_loss": charge >= 0,
            }
        )
    return {
        "format": "denspart-spline-proatom-basis-v1",
        "metadata": {},
        "elements": {
            "1": {
                "symbol": "H",
                "radii": radii.tolist(),
                "radial_weights": radial_weights.tolist(),
                "states": states,
            }
        },
    }


def test_load_lisa_basis_formats():
    legacy = {"1": [[1.0, 2.0], [2.0, 0.5], [0.4, 0.6]]}
    versioned = {
        "format": "denspart-lisa-basis-v1",
        "elements": {"1": {"orders": [1.0, 2.0], "exponents": [2.0, 0.5], "initials": [0.4, 0.6]}},
    }
    legacy_shapes, legacy_initials = load_lisa_basis(legacy)[1]
    versioned_shapes, versioned_initials = load_lisa_basis(versioned)[1]
    assert [(shape.order, shape.exponent) for shape in legacy_shapes] == [
        (shape.order, shape.exponent) for shape in versioned_shapes
    ]
    assert np.array_equal(legacy_initials, versioned_initials)


def test_lisa_periodic_image_and_weight_reconstruction():
    grid = _uniform_periodic_grid()
    center = np.array([0.12, 4.0, 4.0])
    shape = ExponentialShape(2.0, 1.0)
    density = _periodic_density(grid, center, shape)

    # The atom lies near x=0, so its image must also populate the opposite cell face.
    assert np.any(density[grid.points[:, 0] > 7.5] > 0.0)

    result = partition_periodic(
        "lisa",
        center[None, :],
        np.array([1]),
        grid,
        density,
        basis={"1": [[2.0], [1.0], [1.0]]},
        density_cutoff=1.0e-10,
        threshold=1.0e-9,
        inner_threshold=1.0e-10,
        maxiter=20,
    )
    assert result.converged
    assert result.charges[0] == pytest.approx(0.0, abs=2.0e-7)
    assert result.parameters[0][0] == pytest.approx(1.0, abs=2.0e-7)
    assert result.reconstruction_error < 1.0e-12
    valid = result.promolecule > 1.0e-15
    assert np.allclose(result.aim_weights[:, valid].sum(axis=0), 1.0)


def test_lisa_recovers_two_atom_populations():
    grid = _uniform_periodic_grid(length=10.0)
    coordinates = np.array([[2.5, 5.0, 5.0], [7.5, 5.0, 5.0]])
    density = np.zeros(len(grid.points))
    target_coefficients = ((0.7, 0.5), (0.2, 0.6))
    for center, coefficients in zip(coordinates, target_coefficients, strict=True):
        for coefficient, exponent in zip(coefficients, (1.5, 0.4), strict=True):
            shape = ExponentialShape(2.0, exponent)
            localgrid = grid.get_localgrid(center, shape.cutoff(1.0e-10, coefficient))
            radii = np.linalg.norm(localgrid.points - center, axis=1)
            np.add.at(density, localgrid.indices, coefficient * shape.evaluate(radii))

    result = partition_periodic(
        "lisa",
        coordinates,
        np.array([1, 1]),
        grid,
        density,
        basis={"1": [[2.0, 2.0], [1.5, 0.4], [0.5, 0.5]]},
        threshold=1.0e-8,
        inner_threshold=1.0e-9,
        maxiter=100,
    )
    assert result.charges == pytest.approx([-0.2, 0.2], abs=5.0e-6)
    assert np.concatenate(result.parameters) == pytest.approx(
        np.ravel(target_coefficients), abs=5.0e-5
    )


def test_mbis_one_atom_periodic_density():
    grid = _uniform_periodic_grid(npoint=24)
    center = np.array([0.12, 4.0, 4.0])
    # This is the normalized one-electron MBIS initial pro-atom for hydrogen.
    density = _periodic_density(grid, center, ExponentialShape(1.0, 2.0))
    result = partition_periodic(
        "mbis",
        center[None, :],
        np.array([1]),
        grid,
        density,
        density_cutoff=1.0e-10,
        threshold=1.0e-8,
        inner_threshold=1.0e-9,
        maxiter=20,
    )
    assert result.converged
    assert result.charges[0] == pytest.approx(0.0, abs=5.0e-4)
    assert result.reconstruction_error < 1.0e-12


def test_spline_hirshfeld_i_and_avh_use_shared_states():
    basis = _spline_basis()
    states = load_spline_proatoms(basis)[1]
    assert [state.electrons for state in states] == [2, 1, 0]
    neutral = next(state for state in states if state.charge == 0)
    grid = _uniform_periodic_grid(length=12.0, npoint=24)
    center = np.array([0.12, 6.0, 6.0])
    density = _periodic_density(grid, center, neutral)

    hirshfeld_i = partition_periodic(
        "hirshfeld-i",
        center[None, :],
        np.array([1]),
        grid,
        density,
        basis=basis,
        threshold=1.0e-8,
        maxiter=20,
    )
    avh = partition_periodic(
        "avh-b",
        center[None, :],
        np.array([1]),
        grid,
        density,
        basis=basis,
        threshold=1.0e-8,
        maxiter=20,
    )
    assert hirshfeld_i.converged
    assert avh.converged
    assert hirshfeld_i.charges[0] == pytest.approx(avh.charges[0], abs=1.0e-7)
    # The synthetic monoanion is marked unbound, so AVH-B retains only neutral H.
    assert len(avh.parameters[0]) == 1
    with pytest.raises(ValueError, match="AVH-A for Z=1 is missing required states"):
        partition_periodic(
            "avh-a",
            center[None, :],
            np.array([1]),
            grid,
            density,
            basis=basis,
        )


def test_periodic_command_line_roundtrip(tmp_path):
    grid = _uniform_periodic_grid()
    center = np.array([0.12, 4.0, 4.0])
    density = _periodic_density(grid, center, ExponentialShape(2.0, 1.0))
    input_file = tmp_path / "density.npz"
    output_file = tmp_path / "partition.npz"
    np.savez(
        input_file,
        atcoords=center[None, :],
        atnums=np.array([1]),
        points=grid.points,
        weights=grid.weights,
        density=density,
        cellvecs=np.eye(3) * 8.0,
    )
    basis_file = tmp_path / "basis.json"
    basis_file.write_text('{"1": [[2.0], [1.0], [1.0]]}', encoding="utf8")

    assert (
        periodic_main(
            [
                str(input_file),
                str(output_file),
                "--method",
                "lisa",
                "--basis",
                str(basis_file),
                "--no-aim-weights",
            ]
        )
        == 0
    )
    with np.load(output_file) as result:
        assert "aim_weights" not in result
        assert result["method"] == "lisa"
        assert result["parameter_labels"].tolist() == ["basis_0"]
        assert result["charges"][0] == pytest.approx(0.0, abs=2.0e-7)
