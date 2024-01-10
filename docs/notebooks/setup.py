"""Helper functions for examples."""

import numpy as np
from gbasis.evals.eval import evaluate_basis
from gbasis.wrappers import from_iodata
from grid import BeckeWeights, ExpRTransform, MolGrid, UniformInteger
from iodata import load_one

np.set_printoptions(precision=3, linewidth=np.inf, suppress=True)


def prepare_grid_and_dens(filename):
    """Prepare molecular grid and density."""
    mol = load_one(filename)

    # Specify the integration grid
    rtf = ExpRTransform(5e-4, 2e1, 120 - 1)
    uniform_grid = UniformInteger(120)
    rgrid = rtf.transform_1d_grid(uniform_grid)
    becke = BeckeWeights()
    grid = MolGrid.from_preset(
        mol.atnums, mol.atcoords, rgrid, "fine", becke, rotate=False, store=True
    )

    # Get the spin-summed density matrix
    one_rdm = mol.one_rdms.get("post_scf", mol.one_rdms.get("scf"))
    basis, coord_types = from_iodata(mol)
    basis_grid = evaluate_basis(basis, grid.points, coord_type=coord_types)
    rho = np.einsum("ab,bp,ap->p", one_rdm, basis_grid, basis_grid, optimize=True)
    nelec = grid.integrate(rho)
    print(f"The number of electrons: {nelec}")
    print(f"Coordinates of the atoms \n {mol.atcoords}")
    print(f"Atomic numbers of the atom \n {mol.atnums}")
    return mol, grid, rho


def prepare_argument_dict(mol, grid, rho):
    """Prepare basic input arguments for all AIM methods."""
    kwargs = {
        "coordinates": mol.atcoords,
        "numbers": mol.atnums,
        "pseudo_numbers": mol.atnums,
        "grid": grid,
        "moldens": rho,
        "lmax": 3,
        "maxiter": 1000,
        "threshold": 1e-6,
    }
    return kwargs


def print_results(part):
    """Print partitioning results."""
    print("charges:")
    print(part.cache["charges"])
    print("cartesian multipoles:")
    print(part.cache["cartesian_multipoles"])
