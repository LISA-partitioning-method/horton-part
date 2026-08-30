# HORTON-PART: molecular and periodic density partitioning.
# Copyright (C) 2023-2026 The HORTON-PART Development Team
#
# This file is part of HORTON-PART and is distributed under the GNU General
# Public License, version 3 or (at your option) any later version.
"""Basis loaders and radial shapes for periodic stockholder partitioning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.interpolate import CubicSpline

from ..core.basis import evaluate_function

__all__ = [
    "ExponentialShape",
    "RadialSplineState",
    "load_lisa_basis",
    "load_spline_proatoms",
]

LISA_BASIS_FORMATS = frozenset({"aim-lisa-basis-v1", "denspart-lisa-basis-v1"})
SPLINE_PROATOM_FORMATS = frozenset(
    {"aim-proatom-spline-v1", "denspart-spline-proatom-basis-v1"}
)


def _load_mapping(source):
    """Return a JSON-compatible mapping loaded from ``source``."""
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf8") as handle:
            source = json.load(handle)
    if not isinstance(source, Mapping):
        raise TypeError("A basis must be a mapping or a path to a JSON mapping.")
    return source


@dataclass(frozen=True)
class ExponentialShape:
    """Normalized spherical function proportional to ``exp(-alpha r**order)``."""

    order: float
    exponent: float

    def __post_init__(self):
        if not np.isfinite(self.order) or self.order <= 0.0:
            raise ValueError("Exponential orders must be finite and positive.")
        if not np.isfinite(self.exponent) or self.exponent <= 0.0:
            raise ValueError("Exponential exponents must be finite and positive.")

    def evaluate(self, radii):
        """Evaluate the unit-integral radial shape."""
        return evaluate_function(self.order, 1.0, self.exponent, np.asarray(radii))

    def cutoff(self, density_cutoff, scale=1.0):
        """Return a radius beyond which the scaled shape is below a cutoff."""
        if density_cutoff <= 0.0:
            return np.inf
        if scale <= 0.0:
            return 0.0
        value_at_origin = float(self.evaluate(np.zeros(1))[0]) * scale
        if value_at_origin <= density_cutoff:
            return 0.0
        return float(
            (np.log(value_at_origin / density_cutoff) / self.exponent) ** (1.0 / self.order)
        )


@dataclass(frozen=True)
class RadialSplineState:
    """One isolated atomic state represented by a unit-integral radial spline."""

    charge: int
    electrons: int
    radii: np.ndarray
    radial_weights: np.ndarray
    density: np.ndarray
    energy_hartree: float | None = None
    bound_to_electron_loss: bool | None = None

    def __post_init__(self):
        radii = np.asarray(self.radii, dtype=float)
        radial_weights = np.asarray(self.radial_weights, dtype=float)
        density = np.asarray(self.density, dtype=float)
        if (
            radii.ndim != 1
            or len(radii) < 2
            or radial_weights.shape != radii.shape
            or density.shape != radii.shape
        ):
            raise ValueError("Spline radii and densities must be matching one-dimensional arrays.")
        if radii[0] < 0.0 or not np.all(np.diff(radii) > 0.0):
            raise ValueError("Spline radii must be nonnegative and strictly increasing.")
        if not np.isfinite(density).all() or np.any(density < 0.0):
            raise ValueError("Spline densities must be finite and nonnegative.")
        if self.electrons < 0:
            raise ValueError("Atomic spline states cannot have a negative electron count.")
        if radii[0] > 0.0:
            spline_radii = np.concatenate(([0.0], radii))
            spline_density = np.concatenate(([density[0]], density))
        else:
            spline_radii = radii
            spline_density = density
        spline = CubicSpline(
            spline_radii,
            spline_density,
            bc_type=((1, 0.0), "natural"),
            extrapolate=False,
        )
        object.__setattr__(self, "radii", radii)
        object.__setattr__(self, "radial_weights", radial_weights)
        object.__setattr__(self, "density", density)
        object.__setattr__(self, "_spline", spline)

    def evaluate(self, radii):
        """Evaluate the unit-integral state shape at radial distances."""
        values = self._spline(np.asarray(radii, dtype=float))
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(values, 0.0, np.inf)

    def cutoff(self, density_cutoff, scale=1.0):
        """Return the tabulated radius needed for a scaled density cutoff."""
        if density_cutoff <= 0.0:
            return float(self.radii[-1])
        if scale <= 0.0:
            return 0.0
        active = np.flatnonzero(scale * self.density >= density_cutoff)
        if not len(active):
            return 0.0
        index = min(int(active[-1]) + 1, len(self.radii) - 1)
        return float(self.radii[index])


def load_lisa_basis(source):
    """Load legacy HORTON-Part or versioned LISA exponential basis data."""
    raw = _load_mapping(source)
    if raw.get("format") in LISA_BASIS_FORMATS:
        raw = raw.get("elements", {})
    result = {}
    for raw_number, element in raw.items():
        try:
            number = int(raw_number)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid atomic number in LISA basis: {raw_number!r}.") from exc
        if isinstance(element, Mapping):
            orders = element.get("orders")
            exponents = element.get("exponents")
            initials = element.get("initials")
        else:
            if not isinstance(element, (list, tuple)) or len(element) != 3:
                raise ValueError(f"Invalid LISA basis entry for atomic number {number}.")
            orders, exponents, initials = element
        orders = np.asarray(orders, dtype=float)
        exponents = np.asarray(exponents, dtype=float)
        initials = np.asarray(initials, dtype=float)
        if orders.ndim != 1 or exponents.shape != orders.shape or initials.shape != orders.shape:
            raise ValueError(f"LISA basis arrays for atomic number {number} have unequal shapes.")
        if not np.isfinite(initials).all() or np.any(initials < 0.0) or initials.sum() <= 0.0:
            raise ValueError(f"LISA initials for atomic number {number} must be nonnegative.")
        shapes = tuple(
            ExponentialShape(float(order), float(exponent))
            for order, exponent in zip(orders, exponents, strict=True)
        )
        result[number] = (shapes, initials)
    if not result:
        raise ValueError("The LISA basis contains no elements.")
    return result


def load_spline_proatoms(source, population_tolerance=1.0e-6):
    """Load electron-normalized isolated atomic densities as unit radial shapes."""
    raw = _load_mapping(source)
    if raw.get("format") not in SPLINE_PROATOM_FORMATS:
        names = " or ".join(sorted(SPLINE_PROATOM_FORMATS))
        raise ValueError(f"Expected an {names} mapping.")
    elements = raw.get("elements")
    if not isinstance(elements, Mapping) or not elements:
        raise ValueError("The spline pro-atom basis contains no elements.")
    result = {}
    for raw_number, element in elements.items():
        number = int(raw_number)
        radii = np.asarray(element.get("radii"), dtype=float)
        radial_weights = np.asarray(element.get("radial_weights"), dtype=float)
        if radial_weights.shape != radii.shape or np.any(radial_weights <= 0.0):
            raise ValueError(f"Radial weights for Z={number} must be positive and match radii.")
        volume_weights = 4.0 * np.pi * radii**2 * radial_weights
        states = []
        seen = set()
        for raw_state in element.get("states", []):
            charge = int(raw_state["charge"])
            if charge in seen:
                raise ValueError(f"Duplicate spline charge {charge:+d} for Z={number}.")
            seen.add(charge)
            electrons = int(raw_state.get("electrons", number - charge))
            if electrons != number - charge or electrons < 0:
                raise ValueError(f"Invalid electron count for Z={number}, charge={charge:+d}.")
            density = np.asarray(raw_state.get("density"), dtype=float)
            if density.shape != radii.shape:
                raise ValueError(f"Spline density for Z={number}, charge={charge:+d} is malformed.")
            population = float(np.dot(volume_weights, density))
            if not np.isclose(population, electrons, rtol=0.0, atol=population_tolerance):
                raise ValueError(
                    f"Spline state Z={number}, charge={charge:+d} integrates to "
                    f"{population:.12g}, expected {electrons}."
                )
            states.append(
                RadialSplineState(
                    charge=charge,
                    electrons=electrons,
                    radii=radii.copy(),
                    radial_weights=radial_weights.copy(),
                    density=density / electrons if electrons else np.zeros_like(density),
                    energy_hartree=raw_state.get("energy_hartree"),
                    bound_to_electron_loss=raw_state.get("bound_to_electron_loss"),
                )
            )
        if not states or 0 not in seen:
            raise ValueError(f"Spline basis for Z={number} must contain a neutral state.")
        result[number] = tuple(sorted(states, key=lambda state: state.charge))
    return result
