"""Small, dependency-free periodic examples used by the documentation notebooks."""

import numpy as np
from grid import PeriodicGrid

from horton_part.periodic.basis import ExponentialShape


def make_demo_system(length=10.0, npoint=16):
    """Return a two-hydrogen periodic grid and an asymmetric two-electron density.

    Coordinates and cell vectors are in bohr. The density is assembled from periodic
    images of normalized Gaussian functions and integrates to approximately two electrons.
    """
    cell = np.eye(3) * length
    axis = (np.arange(npoint) + 0.5) / npoint
    fractional = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    points = fractional @ cell
    weights = np.full(len(points), abs(np.linalg.det(cell)) / len(points))
    grid = PeriodicGrid(points, weights, cell, wrap=True)

    coordinates = np.array([[2.5, 5.0, 5.0], [7.5, 5.0, 5.0]])
    numbers = np.array([1, 1])
    density = np.zeros(len(points))
    target_coefficients = ((0.7, 0.5), (0.2, 0.6))
    exponents = (1.5, 0.4)
    for center, coefficients in zip(coordinates, target_coefficients, strict=True):
        for coefficient, exponent in zip(coefficients, exponents, strict=True):
            shape = ExponentialShape(2.0, exponent)
            localgrid = grid.get_localgrid(center, shape.cutoff(1.0e-10, coefficient))
            radii = np.linalg.norm(localgrid.points - center, axis=1)
            np.add.at(density, localgrid.indices, coefficient * shape.evaluate(radii))

    lisa_basis = {"1": [[2.0, 2.0], list(exponents), [0.5, 0.5]]}
    return coordinates, numbers, grid, density, lisa_basis


def make_demo_spline_basis():
    """Return normalized H-, H, and H+ radial states for the stockholder examples."""
    radii = np.linspace(1.0e-4, 12.0, 801)
    radial_weights = np.full_like(radii, radii[1] - radii[0])
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
                "bound_to_electron_loss": True,
            }
        )
    return {
        "format": "aim-proatom-spline-v1",
        "metadata": {"purpose": "executable HORTON-Part documentation example"},
        "elements": {
            "1": {
                "symbol": "H",
                "radii": radii.tolist(),
                "radial_weights": radial_weights.tolist(),
                "states": states,
            }
        },
    }


def print_diagnostics(result):
    """Print the conservation checks readers should perform for a periodic partition."""
    active = result.promolecule > 1.0e-15
    weight_error = np.max(np.abs(result.aim_weights[:, active].sum(axis=0) - 1.0))
    print(f"method: {result.method}; solver: {result.solver}")
    print(f"charges: {np.array2string(result.charges, precision=6)}")
    print(f"sum of charges: {result.charges.sum():+.3e} e")
    print(f"maximum partition-of-unity error: {weight_error:.3e}")
    print(f"density reconstruction error: {result.reconstruction_error:.3e}")
