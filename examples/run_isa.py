#!/usr/bin/env python

import numpy as np
from horton_part import log
from iodata import load_one
from gbasis.evals.eval import evaluate_basis
from gbasis.wrappers import from_iodata
from horton_part import IterativeStockholderWPart
from grid import ExpRTransform, UniformInteger, BeckeWeights, MolGrid
from utils import load_fchk

np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
np.random.seed(44)
log.set_level(0)


def main(name):
    # Load the Gaussian output from file from HORTON's test data directory.
    fn_fchk = load_fchk(name)
    # Replace the previous line with any other fchk file, e.g. fn_fchk = 'yourfile.fchk'.
    mol = load_one(fn_fchk)

    # Specify the integration grid
    rtf = ExpRTransform(5e-4, 2e1, 120 - 1)
    uniform_grid = UniformInteger(120)
    rgrid = rtf.transform_1d_grid(uniform_grid)
    becke = BeckeWeights()
    grid = MolGrid.from_preset(
        mol.atnums, mol.atcoords, rgrid, "fine", becke, rotate=False, store=True
    )

    # # Get the spin-summed density matrix
    one_rdm = mol.one_rdms.get("post_scf", mol.one_rdms.get("scf"))
    basis, coord_types = from_iodata(mol)
    basis_grid = evaluate_basis(basis, grid.points, coord_type=coord_types)
    rho = np.einsum("ab,bp,ap->p", one_rdm, basis_grid, basis_grid, optimize=True)

    nelec = grid.integrate(rho)
    print("nelec = {:.2f}".format(nelec))

    kwargs = {
        "coordinates": mol.atcoords,
        "numbers": mol.atnums,
        "pseudo_numbers": mol.atnums,
        "grid": grid,
        "moldens": rho,
        "lmax": 3,
        "maxiter": 1000,
    }

    part = IterativeStockholderWPart(**kwargs)
    part.do_all()

    print("charges:")
    print(part.cache["charges"])
    print("cartesian multipoles:")
    print(part.cache["cartesian_multipoles"])


if __name__ == "__main__":
    for name in ["co", "clo-"]:
        main(name)
        print("#" * 80)

# CO
"""
charges:
[ 0.198 -0.199]
cartesian multipoles:
[[ 0.198  0.     0.    -0.304 -3.879 -0.     0.    -3.879 -0.    -4.774  0.     0.    -0.235  0.    -0.     0.     0.    -0.235  0.    -1.16 ]
 [-0.199  0.     0.    -0.013 -3.746 -0.     0.    -3.746  0.    -3.838 -0.     0.     0.115  0.    -0.    -0.     0.     0.115  0.     0.136]]
"""

# ClO^-
"""
charges:
[-0.377 -0.623]
cartesian multipoles:
[[ -0.377  -0.      0.     -0.243 -10.517   0.     -0.    -10.517   0.     -9.542   0.      0.     -0.515  -0.     -0.     -0.      0.     -0.515   0.      0.396]
 [ -0.623  -0.     -0.      0.132  -4.666   0.      0.     -4.666   0.     -3.961  -0.     -0.      0.236  -0.     -0.     -0.     -0.      0.236  -0.     -0.075]]
"""

# C5H12
"""
charges:
[ 0.665 -0.473 -0.473 -0.473 -0.473  0.102  0.102  0.102  0.102  0.102  0.102  0.102  0.102  0.102  0.102  0.102  0.102]
cartesian multipoles:
[[ 0.665  0.    -0.    -0.    -2.728 -0.     0.    -2.728 -0.    -2.728  0.    -0.     0.     0.    -0.064  0.    -0.     0.    -0.     0.   ]
 [-0.473 -0.002 -0.002 -0.002 -4.655  0.011  0.011 -4.655  0.011 -4.655 -0.013 -0.005 -0.005 -0.005  0.121 -0.005 -0.013 -0.005 -0.005 -0.013]
 [-0.473  0.002  0.002 -0.002 -4.655  0.011 -0.011 -4.655 -0.011 -4.655  0.013  0.005 -0.005  0.005  0.121  0.005  0.013 -0.005  0.005 -0.013]
 [-0.473  0.002 -0.002  0.002 -4.655 -0.011  0.011 -4.655 -0.011 -4.655  0.013 -0.005  0.005  0.005  0.121  0.005 -0.013  0.005 -0.005  0.013]
 [-0.473 -0.002  0.002  0.002 -4.655 -0.011 -0.011 -4.655  0.011 -4.655 -0.013  0.005  0.005 -0.005  0.121 -0.005  0.013  0.005  0.005  0.013]
 [ 0.102  0.018 -0.016  0.018 -0.611 -0.007  0.009 -0.607 -0.007 -0.611 -0.005 -0.008  0.008  0.007  0.002  0.008  0.     0.007 -0.008 -0.005]
 [ 0.102  0.018  0.018 -0.016 -0.611  0.009 -0.007 -0.611 -0.007 -0.607 -0.005  0.008 -0.008  0.008  0.002  0.007 -0.005 -0.008  0.007  0.   ]
 [ 0.102 -0.016  0.018  0.018 -0.607 -0.007 -0.007 -0.611  0.009 -0.611  0.     0.007  0.007 -0.008  0.002 -0.008 -0.005  0.008  0.008 -0.005]
 [ 0.102 -0.018 -0.018 -0.016 -0.611  0.009  0.007 -0.611  0.007 -0.607  0.005 -0.008 -0.008 -0.008  0.002 -0.007  0.005 -0.008 -0.007  0.   ]
 [ 0.102  0.016 -0.018  0.018 -0.607 -0.007  0.007 -0.611 -0.009 -0.611 -0.    -0.007  0.007  0.008  0.002  0.008  0.005  0.008 -0.008 -0.005]
 [ 0.102 -0.018  0.016  0.018 -0.611 -0.007 -0.009 -0.607  0.007 -0.611  0.005  0.008  0.008 -0.007  0.002 -0.008 -0.     0.007  0.008 -0.005]
 [ 0.102 -0.018 -0.016 -0.018 -0.611  0.007  0.009 -0.607  0.007 -0.611  0.005 -0.008 -0.008 -0.007  0.002 -0.008  0.    -0.007 -0.008  0.005]
 [ 0.102 -0.018  0.018  0.016 -0.611 -0.009 -0.007 -0.611  0.007 -0.607  0.005  0.008  0.008 -0.008  0.002 -0.007 -0.005  0.008  0.007 -0.   ]
 [ 0.102  0.016  0.018 -0.018 -0.607  0.007 -0.007 -0.611 -0.009 -0.611 -0.     0.007 -0.007  0.008  0.002  0.008 -0.005 -0.008  0.008  0.005]
 [ 0.102  0.018 -0.018  0.016 -0.611 -0.009  0.007 -0.611 -0.007 -0.607 -0.005 -0.008  0.008  0.008  0.002  0.007  0.005  0.008 -0.007 -0.   ]
 [ 0.102 -0.016 -0.018 -0.018 -0.607  0.007  0.007 -0.611  0.009 -0.611  0.    -0.007 -0.007 -0.008  0.002 -0.008  0.005 -0.008 -0.008  0.005]
 [ 0.102  0.018  0.016 -0.018 -0.611  0.007 -0.009 -0.607 -0.007 -0.611 -0.005  0.008 -0.008  0.007  0.002  0.008 -0.    -0.007  0.008  0.005]]
"""

# C6H6
"""
charges:
[-0.118 -0.118 -0.119 -0.119 -0.118 -0.119  0.118  0.118  0.118  0.118  0.118  0.118]
cartesian multipoles:
[[-0.118 -0.053  0.011 -0.    -4.364 -0.017 -0.    -4.444 -0.    -4.436 -0.269  0.112 -0.     0.116 -0.    -0.101 -0.08   0.     0.021 -0.   ]
 [-0.118 -0.017  0.051  0.    -4.435 -0.026 -0.    -4.366  0.    -4.432  0.116 -0.066  0.    -0.167 -0.    -0.033  0.215 -0.     0.098  0.   ]
 [-0.119 -0.036 -0.04  -0.    -4.412  0.043 -0.    -4.401  0.    -4.439  0.078 -0.132 -0.    -0.179 -0.    -0.068  0.017 -0.    -0.078 -0.   ]
 [-0.119  0.036  0.041 -0.    -4.411  0.043 -0.    -4.4   -0.    -4.437 -0.078  0.133  0.     0.18  -0.     0.068 -0.016  0.     0.078  0.   ]
 [-0.118  0.017 -0.051  0.    -4.434 -0.026 -0.    -4.365  0.    -4.431 -0.115  0.066  0.     0.167 -0.     0.033 -0.216  0.    -0.098  0.   ]
 [-0.119  0.053 -0.011  0.    -4.365 -0.017 -0.    -4.445  0.    -4.437  0.268 -0.112 -0.    -0.116 -0.     0.101  0.08   0.    -0.021  0.   ]
 [ 0.118  0.042 -0.008 -0.    -0.535  0.001 -0.    -0.529 -0.    -0.563  0.012  0.003 -0.     0.012 -0.     0.008 -0.008 -0.    -0.002 -0.   ]
 [ 0.118  0.014 -0.04  -0.    -0.53   0.002  0.    -0.536 -0.    -0.563  0.012 -0.01  -0.    -0.004 -0.     0.003 -0.014 -0.    -0.008 -0.   ]
 [ 0.118  0.028  0.032 -0.    -0.531 -0.003 -0.    -0.532 -0.    -0.562  0.018  0.    -0.    -0.002  0.     0.005  0.018 -0.     0.006 -0.   ]
 [ 0.118 -0.028 -0.032  0.    -0.531 -0.003 -0.    -0.532 -0.    -0.562 -0.018 -0.     0.     0.002 -0.    -0.005 -0.018  0.    -0.006  0.   ]
 [ 0.118 -0.014  0.04   0.    -0.53   0.002 -0.    -0.536  0.    -0.564 -0.012  0.01   0.     0.004  0.    -0.003  0.014  0.     0.008  0.   ]
 [ 0.118 -0.042  0.008 -0.    -0.535  0.001 -0.    -0.529  0.    -0.563 -0.012 -0.003  0.    -0.012 -0.    -0.008  0.008  0.     0.002  0.   ]]
"""