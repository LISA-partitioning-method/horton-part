# HORTON-PART: molecular and periodic density partitioning.
# Copyright (C) 2023-2026 The HORTON-PART Development Team
#
# This file is part of HORTON-PART and is distributed under the GNU General
# Public License, version 3 or (at your option) any later version.
"""Shared real-space engine for periodic stockholder partitioning."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, SR1, minimize

from ..core.basis import evaluate_function
from ..mbis import get_initial_mbis_propars, opt_mbis_propars
from ..utils import NEGATIVE_CUTOFF

__all__ = [
    "InterpolatedProAtom",
    "LinearProAtom",
    "MBISProAtom",
    "PeriodicPartitionResult",
    "PeriodicStockholder",
]

logger = logging.getLogger(__name__)


@dataclass
class PeriodicPartitionResult:
    """Results returned by a periodic stockholder calculation."""

    method: str
    charges: np.ndarray
    populations: np.ndarray
    grid_populations: np.ndarray
    model_charges: np.ndarray
    model_populations: np.ndarray
    parameters: tuple
    parameter_labels: tuple
    promolecule: np.ndarray
    aim_weights: np.ndarray | None
    iterations: int
    converged: bool
    history: np.ndarray
    reconstruction_error: float

    def to_dict(self):
        """Return arrays suitable for an NPZ result file."""
        result = {
            "method": np.array(self.method),
            "charges": self.charges,
            "populations": self.populations,
            "grid_populations": self.grid_populations,
            "model_charges": self.model_charges,
            "model_populations": self.model_populations,
            "promolecule": self.promolecule,
            "iterations": np.array(self.iterations),
            "converged": np.array(self.converged),
            "history": self.history,
            "reconstruction_error": np.array(self.reconstruction_error),
            "parameter_counts": np.array([len(values) for values in self.parameters], dtype=int),
            "parameters": np.concatenate(self.parameters),
            "parameter_labels": np.concatenate(self.parameter_labels),
        }
        if self.aim_weights is not None:
            result["aim_weights"] = self.aim_weights
        return result


class LinearProAtom:
    """A nonnegative linear expansion in normalized radial shapes."""

    def __init__(self, shapes, coefficients, activation_scale):
        self.shapes = tuple(shapes)
        self.coefficients = np.asarray(coefficients, dtype=float).copy()
        activation_scale = np.asarray(activation_scale, dtype=float)
        if activation_scale.ndim == 0:
            activation_scale = np.full(len(self.shapes), float(activation_scale))
        self.activation_scales = activation_scale.copy()
        if len(self.shapes) == 0 or self.coefficients.shape != (len(self.shapes),):
            raise ValueError("A linear pro-atom requires one coefficient per radial shape.")
        if self.activation_scales.shape != self.coefficients.shape:
            raise ValueError("A linear pro-atom requires one activation scale per radial shape.")
        if np.any(self.coefficients < 0.0) or not np.isfinite(self.coefficients).all():
            raise ValueError("Linear pro-atom coefficients must be finite and nonnegative.")
        if np.any(self.activation_scales < 0.0) or not np.isfinite(self.activation_scales).all():
            raise ValueError("Linear pro-atom activation scales must be finite and nonnegative.")

    @property
    def population(self):
        return float(self.coefficients.sum())

    @property
    def parameters(self):
        return self.coefficients.copy()

    @property
    def parameter_labels(self):
        return np.array(
            [
                (f"charge_{shape.charge:+d}" if hasattr(shape, "charge") else f"basis_{index}")
                for index, shape in enumerate(self.shapes)
            ]
        )

    def basis_values(self, radii):
        return np.asarray([shape.evaluate(radii) for shape in self.shapes])

    def evaluate(self, radii):
        return self.coefficients @ self.basis_values(radii)

    def cutoff(self, density_cutoff, include_inactive=True):
        scales = self.activation_scales if include_inactive else self.coefficients
        return max(
            shape.cutoff(density_cutoff, scale)
            for shape, scale in zip(self.shapes, scales, strict=True)
        )

    def fit(self, density, radii, weights, threshold, maxiter, density_cutoff):
        """Fit nonnegative coefficients to an AIM density with a population constraint."""
        basis = self.basis_values(radii)
        self.fit_basis(density, basis, weights, threshold, maxiter, density_cutoff)

    def fit_basis(self, density, basis, weights, threshold, maxiter, density_cutoff):
        """Fit coefficients using fixed basis functions already evaluated on a grid."""
        basis = np.asarray(basis, dtype=float)
        density = np.asarray(density, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if basis.shape != (len(self.shapes), len(density)) or weights.shape != density.shape:
            raise ValueError(
                "Linear pro-atom basis, density, and weights have incompatible shapes."
            )
        population = float(np.dot(weights, density))
        if population <= 0.0:
            raise RuntimeError("Cannot fit a pro-atom with nonpositive population.")
        initial = np.maximum(self.coefficients, population * 1.0e-10 / len(self.coefficients))
        initial *= population / initial.sum()

        def objective(coefficients):
            proatom = coefficients @ basis
            valid = (density > density_cutoff) & (proatom > density_cutoff)
            ratio = np.divide(
                density,
                proatom,
                out=np.zeros_like(density),
                where=valid,
            )
            value = np.einsum(
                "i,i,i->",
                weights[valid],
                density[valid],
                np.log(density[valid]) - np.log(proatom[valid]),
            )
            gradient = -np.einsum("i,ki->k", weights * ratio, basis)
            return value, gradient

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            jac=True,
            bounds=Bounds(np.zeros_like(initial), np.full_like(initial, np.inf)),
            constraints=LinearConstraint(np.ones((1, len(initial))), population, population),
            options={"ftol": threshold, "maxiter": maxiter, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"Linear pro-atom fit failed: {result.message}")
        self.coefficients[:] = np.clip(result.x, 0.0, np.inf)


class InterpolatedProAtom:
    """A Hirshfeld-I pro-atom interpolated between integer-charge states."""

    def __init__(self, number, states):
        self.number = int(number)
        self.states = {state.charge: state for state in states}
        if 0 not in self.states:
            raise ValueError(f"Hirshfeld-I requires a neutral state for Z={self.number}.")
        self.charge = 0.0

    @property
    def population(self):
        return self.number - self.charge

    @property
    def parameters(self):
        return np.array([self.charge])

    @property
    def parameter_labels(self):
        return np.array(["charge"])

    def interpolation_info(self, charge=None):
        charge = self.charge if charge is None else float(charge)
        minimum, maximum = min(self.states), max(self.states)
        if maximum == self.number - 1:
            maximum = self.number  # The fully stripped, zero-density state is implicit.
        tolerance = 1.0e-10
        if charge < minimum - tolerance or charge > maximum + tolerance:
            raise ValueError(
                f"Hirshfeld-I charge {charge:+.8f} for Z={self.number} is outside "
                f"[{minimum:+d}, {maximum:+d}]."
            )
        charge = float(np.clip(charge, minimum, maximum))
        nearest = int(round(charge))
        if abs(charge - nearest) < tolerance:
            if nearest not in self.states and nearest != self.number:
                raise ValueError(f"Missing Hirshfeld-I state {nearest:+d} for Z={self.number}.")
            return nearest, nearest, 0.0
        lower = int(np.floor(charge))
        upper = lower + 1
        missing = [
            state for state in (lower, upper) if state not in self.states and state != self.number
        ]
        if missing:
            raise ValueError(
                f"Hirshfeld-I needs adjacent states {lower:+d} and {upper:+d} for Z={self.number}."
            )
        return lower, upper, charge - lower

    def evaluate(self, radii):
        lower, upper, fraction = self.interpolation_info()
        density = np.zeros_like(np.asarray(radii, dtype=float))
        if lower in self.states:
            lower_state = self.states[lower]
            density += (1.0 - fraction) * lower_state.electrons * lower_state.evaluate(radii)
        if upper != lower and upper in self.states:
            upper_state = self.states[upper]
            density += fraction * upper_state.electrons * upper_state.evaluate(radii)
        return density

    def cutoff(self, density_cutoff, include_inactive=False):
        lower, upper, fraction = self.interpolation_info()
        states = []
        if lower in self.states:
            states.append((self.states[lower], 1.0 - fraction))
        if upper != lower and upper in self.states:
            states.append((self.states[upper], fraction))
        return max(
            (
                state.cutoff(density_cutoff, max(weight * state.electrons, 1.0e-12))
                for state, weight in states
            ),
            default=0.0,
        )


class MBISProAtom:
    """Minimal Slater-shell pro-atom used by periodic MBIS."""

    def __init__(self, number):
        self.number = int(number)
        self.propars = get_initial_mbis_propars(self.number)

    @property
    def population(self):
        return float(self.propars[::2].sum())

    @property
    def parameters(self):
        return self.propars.copy()

    @property
    def parameter_labels(self):
        labels = []
        for ishell in range(len(self.propars) // 2):
            labels.extend((f"population_{ishell}", f"exponent_{ishell}"))
        return np.array(labels)

    def evaluate(self, radii):
        result = np.zeros_like(radii, dtype=float)
        for population, exponent in self.propars.reshape(-1, 2):
            result += self.evaluate_shell(population, exponent, radii)
        return result

    @staticmethod
    def evaluate_shell(population, exponent, radii):
        """Evaluate one normalized Slater shell."""
        return evaluate_function(1.0, population, exponent, np.asarray(radii))

    @staticmethod
    def shell_derivatives(population, exponent, radii):
        """Return derivatives of one shell with respect to population and exponent."""
        radii = np.asarray(radii)
        exponential = np.exp(-exponent * radii)
        population_derivative = exponent**3 * exponential / (8.0 * np.pi)
        exponent_derivative = (
            population * exponent**2 * (3.0 - exponent * radii) * exponential / (8.0 * np.pi)
        )
        return population_derivative, exponent_derivative

    @staticmethod
    def shell_cutoff(population, exponent, density_cutoff):
        """Return the fixed local-grid radius for one shell."""
        if density_cutoff <= 0.0:
            return np.inf
        value_at_origin = population * exponent**3 / (8.0 * np.pi)
        if value_at_origin <= density_cutoff:
            return 0.0
        return np.log(value_at_origin / density_cutoff) / exponent

    def cutoff(self, density_cutoff, include_inactive=False):
        radii = []
        for population, exponent in self.propars.reshape(-1, 2):
            if population <= 0.0:
                continue
            value_at_origin = population * exponent**3 / (8.0 * np.pi)
            if density_cutoff <= 0.0:
                return np.inf
            if value_at_origin > density_cutoff:
                radii.append(np.log(value_at_origin / density_cutoff) / exponent)
        return max(radii, default=0.0)

    def fit(self, density, radii, weights, threshold, density_cutoff):
        self.propars[:] = opt_mbis_propars(
            density,
            self.propars.copy(),
            weights,
            radii,
            threshold,
            density_cutoff=density_cutoff,
            logger=logger,
        )


class PeriodicStockholder:
    """Common stockholder iteration on a grid supporting periodic local grids."""

    def __init__(
        self,
        coordinates,
        numbers,
        grid,
        density,
        models,
        pseudo_numbers=None,
        density_cutoff=1.0e-10,
    ):
        self.coordinates = np.asarray(coordinates, dtype=float)
        self.numbers = np.asarray(numbers, dtype=int)
        self.pseudo_numbers = (
            self.numbers.astype(float)
            if pseudo_numbers is None
            else np.asarray(pseudo_numbers, dtype=float)
        )
        self.grid = grid
        self.density = np.asarray(density, dtype=float)
        self.models = tuple(models)
        self.density_cutoff = float(density_cutoff)
        self._linear_basis_cache = {}
        self._mbis_shell_cache = None
        self._validate()

    def _validate(self):
        natom = len(self.numbers)
        if natom == 0:
            raise ValueError("At least one atom is required.")
        if self.coordinates.shape != (natom, 3):
            raise ValueError("Coordinates must have shape (natom, 3).")
        if not np.isfinite(self.coordinates).all():
            raise ValueError("Coordinates must be finite.")
        if np.any(self.numbers <= 0):
            raise ValueError("Atomic numbers must be positive.")
        if self.pseudo_numbers.shape != (natom,) or len(self.models) != natom:
            raise ValueError("Numbers, pseudo-numbers, models, and coordinates must agree.")
        if not np.isfinite(self.pseudo_numbers).all() or np.any(self.pseudo_numbers < 0.0):
            raise ValueError("Pseudo-numbers must be finite and nonnegative.")
        for attr in ("points", "weights", "get_localgrid"):
            if not hasattr(self.grid, attr):
                raise TypeError(f"The grid does not provide the required {attr!r} interface.")
        points = np.asarray(self.grid.points)
        weights = np.asarray(self.grid.weights)
        if points.shape != (len(weights), 3):
            raise ValueError("Periodic grid points must have shape (npoint, 3).")
        if (
            not np.isfinite(points).all()
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or not np.any(weights > 0.0)
        ):
            raise ValueError("Periodic grid points and nonnegative weights must be finite.")
        if self.density.shape != weights.shape:
            raise ValueError("Density and grid weights must have the same shape.")
        if not np.isfinite(self.density).all() or np.any(self.density < NEGATIVE_CUTOFF):
            raise ValueError(
                f"The electron density must be finite and no smaller than {NEGATIVE_CUTOFF:.1e}."
            )
        if np.any(self.density < 0.0):
            self.density = np.maximum(self.density, 0.0)
        if not np.dot(weights, self.density) > 0.0:
            raise ValueError("The integrated electron density must be positive.")
        if not np.isfinite(self.density_cutoff) or self.density_cutoff <= 0.0:
            raise ValueError("Periodic calculations require a finite positive density cutoff.")

    def _localgrid(self, iatom, include_inactive=False):
        radius = self.models[iatom].cutoff(self.density_cutoff, include_inactive)
        if not np.isfinite(radius):
            raise ValueError("Periodic calculations require finite pro-atom cutoff radii.")
        return self.grid.get_localgrid(self.coordinates[iatom], radius)

    def _evaluate_local(self, iatom, localgrid):
        radii = np.linalg.norm(localgrid.points - self.coordinates[iatom], axis=1)
        return self.models[iatom].evaluate(radii), radii

    def _linear_basis(self, iatom):
        """Evaluate fixed linear shapes as periodic sums on the primitive-cell grid."""
        if iatom not in self._linear_basis_cache:
            model = self.models[iatom]
            if not isinstance(model, LinearProAtom):
                raise TypeError("Periodic linear bases are only defined for LinearProAtom models.")
            basis = np.zeros((len(model.shapes), len(self.density)))
            for ibasis, (shape, scale) in enumerate(
                zip(model.shapes, model.activation_scales, strict=True)
            ):
                radius = shape.cutoff(self.density_cutoff, scale)
                localgrid = self.grid.get_localgrid(self.coordinates[iatom], radius)
                radii = np.linalg.norm(localgrid.points - self.coordinates[iatom], axis=1)
                np.add.at(basis[ibasis], localgrid.indices, shape.evaluate(radii))
            self._linear_basis_cache[iatom] = basis
        return self._linear_basis_cache[iatom]

    def _linear_proatom(self, iatom):
        model = self.models[iatom]
        return model.coefficients @ self._linear_basis(iatom)

    def _mbis_shells(self):
        """Return fixed local grids and radii for all MBIS shells."""
        if self._mbis_shell_cache is None:
            shells = []
            for iatom, model in enumerate(self.models):
                if not isinstance(model, MBISProAtom):
                    continue
                for ishell, (population, exponent) in enumerate(model.propars.reshape(-1, 2)):
                    radius = model.shell_cutoff(population, exponent, self.density_cutoff)
                    localgrid = self.grid.get_localgrid(self.coordinates[iatom], radius)
                    radii = np.linalg.norm(localgrid.points - self.coordinates[iatom], axis=1)
                    shells.append((iatom, ishell, radius, localgrid, radii))
            self._mbis_shell_cache = tuple(shells)
        return self._mbis_shell_cache

    def _expand_mbis_shells(self):
        """Expand cached shell grids when optimized functions extend beyond them."""
        expanded = False
        shells = []
        for iatom, ishell, radius, localgrid, radii in self._mbis_shells():
            population, exponent = self.models[iatom].propars.reshape(-1, 2)[ishell]
            required = self.models[iatom].shell_cutoff(
                population,
                exponent,
                self.density_cutoff,
            )
            if required > radius * (1.0 + 1.0e-8):
                radius = required
                localgrid = self.grid.get_localgrid(self.coordinates[iatom], radius)
                radii = np.linalg.norm(localgrid.points - self.coordinates[iatom], axis=1)
                expanded = True
            shells.append((iatom, ishell, radius, localgrid, radii))
        self._mbis_shell_cache = tuple(shells)
        return expanded

    def _mbis_proatoms(self):
        """Evaluate MBIS pro-atoms using the fixed per-shell local grids."""
        proatoms = np.zeros((len(self.models), len(self.density)))
        for iatom, ishell, _, localgrid, radii in self._mbis_shells():
            population, exponent = self.models[iatom].propars.reshape(-1, 2)[ishell]
            values = self.models[iatom].evaluate_shell(population, exponent, radii)
            np.add.at(proatoms[iatom], localgrid.indices, values)
        return proatoms

    def promolecule(self, include_inactive=False):
        if all(isinstance(model, MBISProAtom) for model in self.models):
            return self._mbis_proatoms().sum(axis=0)
        result = np.zeros_like(self.density)
        for iatom in range(len(self.models)):
            if isinstance(self.models[iatom], LinearProAtom):
                result += self._linear_proatom(iatom)
            else:
                localgrid = self._localgrid(iatom, include_inactive)
                values, _ = self._evaluate_local(iatom, localgrid)
                np.add.at(result, localgrid.indices, values)
        return result

    def stockholder_populations(self, promolecule):
        valid = (self.density > 1.0e-15) & (promolecule > 1.0e-15)
        ratio = np.divide(
            self.density,
            promolecule,
            out=np.zeros_like(self.density),
            where=valid,
        )
        if all(isinstance(model, MBISProAtom) for model in self.models):
            return np.einsum("ap,p,p->a", self._mbis_proatoms(), ratio, self.grid.weights), ratio
        populations = np.zeros(len(self.models))
        for iatom in range(len(self.models)):
            if isinstance(self.models[iatom], LinearProAtom):
                populations[iatom] = np.dot(self.grid.weights, self._linear_proatom(iatom) * ratio)
            else:
                localgrid = self._localgrid(iatom)
                proatom, _ = self._evaluate_local(iatom, localgrid)
                populations[iatom] = localgrid.integrate(proatom, ratio[localgrid.indices])
        return populations, ratio

    def _fit_models(self, promolecule, threshold, inner_maxiter, linear):
        valid = (self.density > 1.0e-15) & (promolecule > 1.0e-15)
        ratio = np.divide(
            self.density,
            promolecule,
            out=np.zeros_like(self.density),
            where=valid,
        )
        for iatom, model in enumerate(self.models):
            if linear:
                atom_density = self._linear_proatom(iatom) * ratio
                model.fit_basis(
                    atom_density,
                    self._linear_basis(iatom),
                    self.grid.weights,
                    threshold,
                    inner_maxiter,
                    self.density_cutoff,
                )
            else:
                localgrid = self._localgrid(iatom)
                proatom, radii = self._evaluate_local(iatom, localgrid)
                atom_density = proatom * ratio[localgrid.indices]
                model.fit(
                    atom_density,
                    radii,
                    localgrid.weights,
                    threshold,
                    self.density_cutoff,
                )

    def run_fixed(self, method, return_weights=True):
        promolecule = self.promolecule()
        return self._finalize(method, promolecule, 0, True, [], return_weights)

    def run_hirshfeld_i(self, threshold, maxiter, mixing=0.5, return_weights=True):
        if threshold <= 0.0 or maxiter <= 0:
            raise ValueError("The threshold and maximum iteration count must be positive.")
        if not 0.0 < mixing <= 1.0:
            raise ValueError("Hirshfeld-I mixing must lie in (0, 1].")
        history = [np.array([model.charge for model in self.models])]
        previous_step = None
        use_mixing = False
        for iteration in range(1, maxiter + 1):
            promolecule = self.promolecule()
            populations, _ = self.stockholder_populations(promolecule)
            old = np.array([model.charge for model in self.models])
            raw = self.pseudo_numbers - populations
            for model, charge in zip(self.models, raw, strict=True):
                model.interpolation_info(charge)
            step = raw - old
            if previous_step is not None and np.dot(step, previous_step) < 0.0:
                use_mixing = True
            new = old + (mixing * step if use_mixing else step)
            for model, charge in zip(self.models, new, strict=True):
                model.charge = float(charge)
            previous_step = new - old
            history.append(new.copy())
            if np.max(np.abs(new - old)) < threshold:
                promolecule = self.promolecule()
                return self._finalize(
                    "hirshfeld-i",
                    promolecule,
                    iteration,
                    True,
                    history,
                    return_weights,
                )
        raise RuntimeError(f"Periodic Hirshfeld-I did not converge in {maxiter} iterations.")

    def run_iterative(
        self,
        method,
        threshold,
        inner_threshold,
        maxiter,
        inner_maxiter,
        linear,
        return_weights=True,
    ):
        if threshold <= 0.0 or inner_threshold <= 0.0:
            raise ValueError("Outer and inner thresholds must be positive.")
        if maxiter <= 0 or inner_maxiter <= 0:
            raise ValueError("Outer and inner maximum iteration counts must be positive.")
        history = []
        previous = self.promolecule(include_inactive=linear)
        history.append(self.pseudo_numbers - self.stockholder_populations(previous)[0])
        for iteration in range(1, maxiter + 1):
            self._fit_models(previous, inner_threshold, inner_maxiter, linear)
            current = self.promolecule(include_inactive=linear)
            difference = np.sqrt(np.dot(self.grid.weights, (current - previous) ** 2))
            history.append(self.pseudo_numbers - self.stockholder_populations(current)[0])
            if difference < threshold:
                current = self.promolecule(include_inactive=False)
                return self._finalize(
                    method,
                    current,
                    iteration,
                    True,
                    history,
                    return_weights,
                )
            previous = current
        raise RuntimeError(f"Periodic {method} did not converge in {maxiter} iterations.")

    def run_variational_linear(self, method, threshold, maxiter, return_weights=True):
        """Optimize all linear pro-atom coefficients through the global extended KLD."""
        if threshold <= 0.0 or maxiter <= 0:
            raise ValueError("The threshold and maximum iteration count must be positive.")
        if not all(isinstance(model, LinearProAtom) for model in self.models):
            raise TypeError("Global linear optimization requires LinearProAtom models.")
        basis = np.concatenate(
            [self._linear_basis(iatom) for iatom in range(len(self.models))], axis=0
        )
        initial = np.concatenate([model.coefficients for model in self.models])
        population = float(np.dot(self.grid.weights, self.density))
        numerical_cutoff = min(self.density_cutoff, 1.0e-15)

        def objective(coefficients):
            promolecule = coefficients @ basis
            valid = (self.density > numerical_cutoff) & (promolecule > numerical_cutoff)
            ratio = np.divide(
                self.density,
                promolecule,
                out=np.zeros_like(self.density),
                where=valid,
            )
            kld = np.dot(
                self.grid.weights[valid] * self.density[valid],
                np.log(self.density[valid]) - np.log(promolecule[valid]),
            )
            value = kld - population + coefficients.sum()
            gradient = 1.0 - basis @ (self.grid.weights * ratio)
            return value, gradient

        if method.startswith("avh-"):
            result = minimize(
                objective,
                initial,
                method="SLSQP",
                jac=True,
                bounds=Bounds(np.zeros_like(initial), np.full_like(initial, np.inf)),
                options={"ftol": threshold, "maxiter": maxiter, "disp": False},
            )
        else:
            result = minimize(
                objective,
                initial,
                method="trust-constr",
                jac=True,
                hess=SR1(),
                bounds=Bounds(
                    np.zeros_like(initial),
                    np.full_like(initial, np.inf),
                    keep_feasible=True,
                ),
                options={"gtol": threshold, "maxiter": maxiter},
            )
        if not result.success:
            raise RuntimeError(f"Periodic {method} optimization failed: {result.message}")
        optimized = result.x
        if method.startswith("avh-"):
            optimized_population = float(optimized.sum())
            if optimized_population <= 0.0:
                raise RuntimeError(f"Periodic {method} optimized a nonpositive population.")
            optimized = optimized * (population / optimized_population)
        offset = 0
        for model in self.models:
            count = len(model.coefficients)
            model.coefficients[:] = np.clip(optimized[offset : offset + count], 0.0, np.inf)
            offset += count
        promolecule = self.promolecule()
        initial_populations = []
        offset = 0
        for model in self.models:
            count = len(model.coefficients)
            initial_populations.append(initial[offset : offset + count].sum())
            offset += count
        history = [
            self.pseudo_numbers - np.asarray(initial_populations),
            self.pseudo_numbers - np.array([model.population for model in self.models]),
        ]
        return self._finalize(
            method,
            promolecule,
            int(result.nit),
            True,
            history,
            return_weights,
        )

    def run_variational_mbis(self, threshold, maxiter, return_weights=True):
        """Optimize all MBIS shells globally on fixed per-shell local grids."""
        if threshold <= 0.0 or maxiter <= 0:
            raise ValueError("The threshold and maximum iteration count must be positive.")
        if not all(isinstance(model, MBISProAtom) for model in self.models):
            raise TypeError("Global MBIS optimization requires MBISProAtom models.")

        original_initial = np.concatenate([model.propars for model in self.models])
        population = float(np.dot(self.grid.weights, self.density))
        total_iterations = 0
        for _ in range(10):
            initial = np.concatenate([model.propars for model in self.models])
            shell_data = self._mbis_shells()

            def objective(parameters):
                promolecule = np.zeros_like(self.density)
                for iparam, (_, _, _, localgrid, radii) in enumerate(shell_data):
                    shell_population, exponent = parameters[2 * iparam : 2 * iparam + 2]
                    values = MBISProAtom.evaluate_shell(shell_population, exponent, radii)
                    np.add.at(promolecule, localgrid.indices, values)

                valid = (self.density > 1.0e-15) & (promolecule > 1.0e-15)
                ratio = np.divide(
                    self.density,
                    promolecule,
                    out=np.zeros_like(self.density),
                    where=valid,
                )
                kld = np.dot(
                    self.grid.weights[valid] * self.density[valid],
                    np.log(self.density[valid]) - np.log(promolecule[valid]),
                )
                value = kld - population + parameters[::2].sum()
                gradient = np.zeros_like(parameters)
                for iparam, (_, _, _, localgrid, radii) in enumerate(shell_data):
                    shell_population, exponent = parameters[2 * iparam : 2 * iparam + 2]
                    derivatives = MBISProAtom.shell_derivatives(shell_population, exponent, radii)
                    weighted_ratio = localgrid.weights * ratio[localgrid.indices]
                    gradient[2 * iparam] = 1.0 - np.dot(weighted_ratio, derivatives[0])
                    gradient[2 * iparam + 1] = -np.dot(weighted_ratio, derivatives[1])
                return value, gradient

            lower = np.tile((5.0e-5, 0.1), len(shell_data))
            upper = np.tile((100.0, 1000.0), len(shell_data))
            result = minimize(
                objective,
                initial,
                method="trust-constr",
                jac=True,
                hess=SR1(),
                bounds=Bounds(lower, upper, keep_feasible=True),
                options={"gtol": threshold, "maxiter": maxiter},
            )
            total_iterations += int(result.nit)
            if not result.success:
                raise RuntimeError(f"Periodic MBIS optimization failed: {result.message}")

            offset = 0
            for model in self.models:
                count = len(model.propars)
                model.propars[:] = result.x[offset : offset + count]
                offset += count
            if not self._expand_mbis_shells():
                break
        else:
            raise RuntimeError("Periodic MBIS local grids did not stabilize after 10 expansions.")

        offset = 0
        initial_populations = []
        for model in self.models:
            count = len(model.propars)
            initial_populations.append(original_initial[offset : offset + count : 2].sum())
            offset += count
        history = [
            self.pseudo_numbers - np.asarray(initial_populations),
            self.pseudo_numbers - np.array([model.population for model in self.models]),
        ]
        return self._finalize(
            "mbis",
            self.promolecule(),
            total_iterations,
            True,
            history,
            return_weights,
        )

    def _finalize(
        self,
        method,
        promolecule,
        iterations,
        converged,
        history,
        return_weights,
    ):
        grid_populations, _ = self.stockholder_populations(promolecule)
        model_populations = np.array([model.population for model in self.models])
        populations = grid_populations
        charges = self.pseudo_numbers - grid_populations
        model_charges = self.pseudo_numbers - model_populations
        aim_weights = None
        reconstruction_error = np.nan
        if return_weights:
            aim_weights = np.zeros((len(self.models), len(self.density)))
            valid = promolecule > 1.0e-15
            mbis_proatoms = (
                self._mbis_proatoms()
                if all(isinstance(model, MBISProAtom) for model in self.models)
                else None
            )
            for iatom in range(len(self.models)):
                if mbis_proatoms is not None:
                    proatom = mbis_proatoms[iatom]
                    aim_weights[iatom] = np.divide(
                        proatom,
                        promolecule,
                        out=np.zeros_like(proatom),
                        where=valid,
                    )
                elif isinstance(self.models[iatom], LinearProAtom):
                    proatom = self._linear_proatom(iatom)
                    aim_weights[iatom] = np.divide(
                        proatom,
                        promolecule,
                        out=np.zeros_like(proatom),
                        where=valid,
                    )
                else:
                    localgrid = self._localgrid(iatom)
                    proatom, _ = self._evaluate_local(iatom, localgrid)
                    values = np.divide(
                        proatom,
                        promolecule[localgrid.indices],
                        out=np.zeros_like(proatom),
                        where=valid[localgrid.indices],
                    )
                    np.add.at(aim_weights[iatom], localgrid.indices, values)
            if np.any(valid):
                reconstruction_error = float(
                    np.max(np.abs(aim_weights[:, valid].sum(axis=0) - 1.0))
                )
        return PeriodicPartitionResult(
            method=method,
            charges=charges,
            populations=populations,
            grid_populations=grid_populations,
            model_charges=model_charges,
            model_populations=model_populations,
            parameters=tuple(model.parameters for model in self.models),
            parameter_labels=tuple(model.parameter_labels for model in self.models),
            promolecule=promolecule,
            aim_weights=aim_weights,
            iterations=iterations,
            converged=converged,
            history=np.asarray(history if history else [charges]),
            reconstruction_error=reconstruction_error,
        )
