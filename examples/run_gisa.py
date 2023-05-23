#!/usr/bin/env python

import numpy as np
from horton_grid import BeckeMolGrid, log
from iodata import load_one
from gbasis.evals.eval import evaluate_basis
from gbasis.wrappers import from_iodata
from horton_part import GaussianIterativeStockholderWPart
from utils import load_fchk

np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
np.random.seed(44)

# Load the Gaussian output from file from HORTON's test data directory.
# fn_fchk = context.get_fn("test/water_sto3g_hf_g03.fchk")
# Replace the previous line with any other fchk file, e.g. fn_fchk = 'yourfile.fchk'.
# mol = load_one(fn_fchk)
name = "co"
fn_fchk = load_fchk(name)
mol = load_one(fn_fchk)

# Specify the integration grid
grid = BeckeMolGrid(mol.atcoords, mol.atnums, mol.atnums, mode="keep")

# # Get the spin-summed density matrix
one_rdm = mol.one_rdms.get("post_scf", mol.one_rdms.get("scf"))
basis, coord_types = from_iodata(mol)
basis_grid = evaluate_basis(basis, grid.points, coord_type=coord_types)
rho = np.einsum("ab,bp,ap->p", one_rdm, basis_grid, basis_grid, optimize=True)

nelec = grid.integrate(rho)
log("nelec = {}".format(nelec))

kwargs = {
    "coordinates": mol.atcoords,
    "numbers": mol.atnums,
    "pseudo_numbers": mol.atnums,
    "grid": grid,
    "moldens": rho,
    "lmax": 3,
    "maxiter": 1000,
}

part = GaussianIterativeStockholderWPart(**kwargs)
part.do_partitioning()
part.do_all()

print("charges:")
print(part.cache["charges"])
print("cartesian multipoles:")
print(part.cache["cartesian_multipoles"])
