# HORTON-PART: molecular and periodic density partitioning.
# Copyright (C) 2023-2026 The HORTON-PART Development Team
#
# This file is part of HORTON-PART and is distributed under the GNU General
# Public License, version 3 or (at your option) any later version.
"""Public periodic stockholder methods."""

from __future__ import annotations

import numpy as np

from ..utils import DATA_PATH
from .basis import load_lisa_basis, load_spline_proatoms
from .core import InterpolatedProAtom, LinearProAtom, MBISProAtom, PeriodicStockholder

__all__ = [
    "PeriodicAVHWPart",
    "PeriodicHirshfeldIWPart",
    "PeriodicHirshfeldWPart",
    "PeriodicLISAWPart",
    "PeriodicMBISWPart",
    "partition_periodic",
]


def _select_avh_states(states, variant, number):
    variant = variant.upper()
    if variant == "SUPPLIED":
        selected = tuple(state for state in states if state.electrons > 0)
        if not selected:
            raise ValueError(f"The supplied AVH basis has no populated states for Z={number}.")
        return selected
    if variant == "A":
        required = set(range(-3, int(number)))
        available = {state.charge for state in states if state.electrons > 0}
        missing = sorted(required - available)
        if missing:
            labels = ", ".join(f"{charge:+d}" for charge in missing)
            raise ValueError(f"AVH-A for Z={number} is missing required states: {labels}.")
        return tuple(state for state in states if state.charge in required)
    if variant == "B":
        required = set(range(0, int(number)))
        available = {state.charge for state in states if state.electrons > 0}
        missing = sorted(required - available)
        if missing:
            labels = ", ".join(f"{charge:+d}" for charge in missing)
            raise ValueError(f"AVH-B for Z={number} is missing required states: {labels}.")
        selected = tuple(
            state
            for state in states
            if state.electrons > 0
            and (
                state.charge >= 0
                or (state.charge == -1 and state.bound_to_electron_loss is not False)
            )
        )
        return selected
    if variant == "M":
        return tuple(state for state in states if state.charge == 0 and state.electrons > 0)
    raise ValueError("AVH variant must be 'A', 'B', 'M', or 'supplied'.")


def partition_periodic(
    method,
    coordinates,
    numbers,
    grid,
    density,
    *,
    basis=None,
    pseudo_numbers=None,
    threshold=1.0e-7,
    inner_threshold=1.0e-9,
    maxiter=1000,
    inner_maxiter=1000,
    density_cutoff=1.0e-10,
    mixing=0.5,
    avh_variant="B",
    solver="optimizer",
    return_weights=True,
):
    """Partition a periodic real-space density with a supported AIM method."""
    canonical = method.lower().replace("_", "-")
    aliases = {
        "h": "hirshfeld",
        "hi": "hirshfeld-i",
        "hirshfeldi": "hirshfeld-i",
        "avh-a": "avh",
        "avh-b": "avh",
        "avh-m": "avh",
        "avh-supplied": "avh",
    }
    if canonical.startswith("avh-"):
        avh_variant = canonical.removeprefix("avh-")
    canonical = aliases.get(canonical, canonical)
    solver = solver.lower().replace("_", "-")
    solver = {
        "opt": "optimizer",
        "self-consistent": "sc",
    }.get(solver, solver)
    if solver not in ("optimizer", "sc"):
        raise ValueError("Periodic solver must be 'optimizer' or 'sc'.")
    if canonical in ("hirshfeld", "hirshfeld-i") and solver != "optimizer":
        raise ValueError("The solver option applies only to periodic LISA, AVH, and MBIS.")
    numbers = np.asarray(numbers, dtype=int)
    pseudo_numbers = numbers.astype(float) if pseudo_numbers is None else pseudo_numbers

    if canonical == "mbis":
        models = [MBISProAtom(number) for number in numbers]
    elif canonical == "lisa":
        basis = DATA_PATH / "gauss.json" if basis is None else basis
        library = load_lisa_basis(basis)
        models = []
        for number, pseudo_number in zip(numbers, pseudo_numbers, strict=True):
            if int(number) not in library:
                raise NotImplementedError(f"No LISA basis is available for Z={number}.")
            shapes, initials = library[int(number)]
            coefficients = initials * (float(pseudo_number) / initials.sum())
            models.append(LinearProAtom(shapes, coefficients, coefficients))
    elif canonical in ("hirshfeld", "hirshfeld-i", "avh"):
        if basis is None:
            raise ValueError(f"Periodic {canonical} requires a radial-spline pro-atom basis.")
        library = load_spline_proatoms(basis)
        models = []
        for number, pseudo_number in zip(numbers, pseudo_numbers, strict=True):
            states = library.get(int(number))
            if states is None:
                raise NotImplementedError(f"No spline pro-atoms are available for Z={number}.")
            if canonical == "hirshfeld":
                neutral = next(state for state in states if state.charge == 0)
                models.append(
                    LinearProAtom((neutral,), (float(pseudo_number),), float(pseudo_number))
                )
            elif canonical == "hirshfeld-i":
                models.append(InterpolatedProAtom(number, states))
            else:
                selected = _select_avh_states(states, avh_variant, int(number))
                if solver == "sc":
                    # Match the finite-system AVH implementation. Multiplicative SC
                    # updates cannot activate a coefficient initialized to zero.
                    coefficients = np.ones(len(selected))
                else:
                    coefficients = np.array(
                        [float(pseudo_number) if state.charge == 0 else 0.0 for state in selected]
                    )
                models.append(LinearProAtom(selected, coefficients, float(pseudo_number) + 3.0))
    else:
        raise ValueError(
            "Unsupported periodic method. Choose Hirshfeld, Hirshfeld-I, MBIS, LISA, or AVH."
        )

    engine = PeriodicStockholder(
        coordinates,
        numbers,
        grid,
        density,
        models,
        pseudo_numbers=pseudo_numbers,
        density_cutoff=density_cutoff,
    )
    if canonical == "hirshfeld":
        return engine.run_fixed(canonical, return_weights)
    if canonical == "hirshfeld-i":
        return engine.run_hirshfeld_i(threshold, maxiter, mixing, return_weights)
    if canonical in ("lisa", "avh"):
        if solver == "sc":
            return engine.run_self_consistent_linear(
                f"avh-{avh_variant.lower()}" if canonical == "avh" else canonical,
                threshold,
                maxiter,
                return_weights,
            )
        return engine.run_variational_linear(
            f"avh-{avh_variant.lower()}" if canonical == "avh" else canonical,
            threshold,
            maxiter,
            return_weights,
        )
    if canonical == "mbis":
        if solver == "sc":
            return engine.run_self_consistent_mbis(
                threshold,
                maxiter,
                return_weights,
            )
        return engine.run_variational_mbis(threshold, maxiter, return_weights)
    return engine.run_iterative(
        canonical,
        threshold,
        inner_threshold,
        maxiter,
        inner_maxiter,
        linear=False,
        return_weights=return_weights,
    )


class _PeriodicWPart:
    """Thin class interface matching the naming of molecular HORTON-Part methods."""

    name = None

    def __init__(self, coordinates, numbers, grid, moldens, **options):
        self.coordinates = np.asarray(coordinates)
        self.numbers = np.asarray(numbers)
        self.grid = grid
        self.moldens = np.asarray(moldens)
        self.options = options
        self.result = None

    def do_partitioning(self):
        self.result = partition_periodic(
            self.name,
            self.coordinates,
            self.numbers,
            self.grid,
            self.moldens,
            **self.options,
        )
        return self.result

    @property
    def charges(self):
        if self.result is None:
            self.do_partitioning()
        return self.result.charges

    @property
    def aim_weights(self):
        if self.result is None:
            self.do_partitioning()
        return self.result.aim_weights


class PeriodicHirshfeldWPart(_PeriodicWPart):
    name = "hirshfeld"


class PeriodicHirshfeldIWPart(_PeriodicWPart):
    name = "hirshfeld-i"


class PeriodicMBISWPart(_PeriodicWPart):
    name = "mbis"


class PeriodicLISAWPart(_PeriodicWPart):
    name = "lisa"


class PeriodicAVHWPart(_PeriodicWPart):
    name = "avh"
