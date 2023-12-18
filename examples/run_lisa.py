#!/usr/bin/env python

import numpy as np
from horton_part import log
from iodata import load_one
from gbasis.evals.eval import evaluate_basis
from gbasis.wrappers import from_iodata
from horton_part import LinearISAWPart
from grid import ExpRTransform, UniformInteger, BeckeWeights, MolGrid

from utils import load_fchk

np.set_printoptions(precision=3, suppress=True, linewidth=np.inf)
np.random.seed(44)
log.set_level(0)


def main(name):
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
        "obj_fn_type": 1,
    }

    part = LinearISAWPart(**kwargs)
    # part.do_partitioning()
    part.do_all()

    print("charges:")
    print(part.cache["charges"])
    print("cartesian multipoles:")
    print(part.cache["cartesian_multipoles"])


if __name__ == "__main__":
    for name in ["co", "clo-"]:
        main(name)
        print("#" * 80)

# obj_func_type = 1 and 2, there is no difference
# CO
# obj_func_type = 1: nb_iterations = 20
# obj_func_type = 2: nb_iterations = 128
"""
charges:
[ 0.199 -0.199]
cartesian multipoles:
[[ 0.199  0.     0.    -0.297 -3.893 -0.     0.    -3.893 -0.    -4.807  0.     0.    -0.186  0.    -0.     0.     0.    -0.186  0.    -1.001]
 [-0.199  0.    -0.    -0.02  -3.733 -0.     0.    -3.733 -0.    -3.835 -0.     0.     0.096 -0.    -0.    -0.     0.     0.096  0.     0.096]]
"""

# ClO^-
# obj_func_type = 1: nb_iterations = 10
# obj_func_type = 2: nb_iterations = 369
"""
Comment: using O parameters as Cl atom proposed in the paper. (This is a bad idea, for Cl atom,
a small exponential coefficient should be used for the more diffusion part.

charges:
[-0.404 -0.597]
cartesian multipoles:
[[ -0.404  -0.      0.     -0.128 -10.622   0.     -0.    -10.622   0.     -9.97   -0.      0.     -0.138  -0.     -0.     -0.      0.     -0.138   0.      2.023]
 [ -0.597  -0.      0.      0.103  -4.56    0.      0.     -4.56    0.     -4.004  -0.     -0.      0.203  -0.     -0.     -0.     -0.      0.203  -0.     -0.268]]
"""

"""
Comment: using C parameters as Cl atom proposed in the paper. (This is a bad idea, for Cl atom,
a small exponential coefficient should be used for the more diffusion part.

charges:
[-0.401 -0.599]
cartesian multipoles:
[[ -0.401  -0.      0.     -0.151 -10.578   0.     -0.    -10.578   0.     -9.843  -0.      0.     -0.279  -0.     -0.     -0.      0.     -0.279   0.      1.449]
 [ -0.599  -0.      0.      0.119  -4.604   0.      0.     -4.604   0.     -4.004  -0.     -0.      0.2    -0.     -0.     -0.     -0.      0.2    -0.     -0.281]]
"""

"""
Comment: the result is computed with new parameters optimized as Toon's paper.

 charges:
[-0.383 -0.617]
cartesian multipoles:
[[ -0.383  -0.      0.     -0.216 -10.504   0.     -0.    -10.504   0.     -9.622  -0.      0.     -0.456  -0.     -0.     -0.      0.     -0.456   0.      0.736]
 [ -0.617  -0.     -0.      0.125  -4.678   0.      0.     -4.678   0.     -3.994  -0.     -0.      0.137  -0.     -0.     -0.     -0.      0.137  -0.     -0.282]]
"""

# C5H12
# obj_func_type = 1: nb_iterations = 399
# obj_func_type = 2: nb_iterations = 400
"""
charges:
[ 0.597 -0.485 -0.485 -0.485 -0.485  0.111  0.111  0.111  0.111  0.111  0.111  0.111  0.111  0.111  0.111  0.111  0.111]
cartesian multipoles:
[[ 0.597  0.     0.    -0.    -2.668  0.     0.    -2.668 -0.    -2.668  0.     0.    -0.     0.    -0.062  0.     0.    -0.     0.    -0.   ]
 [-0.485 -0.007 -0.007 -0.007 -4.953 -0.004 -0.004 -4.953 -0.004 -4.953 -0.036  0.014  0.014  0.014  0.163  0.014 -0.036  0.014  0.014 -0.036]
 [-0.485  0.007  0.007 -0.007 -4.953 -0.004  0.004 -4.953  0.004 -4.953  0.036 -0.014  0.014 -0.014  0.163 -0.014  0.036  0.014 -0.014 -0.036]
 [-0.485  0.007 -0.007  0.007 -4.953  0.004 -0.004 -4.953  0.004 -4.953  0.036  0.014 -0.014 -0.014  0.163 -0.014 -0.036 -0.014  0.014  0.036]
 [-0.485 -0.007  0.007  0.007 -4.953  0.004  0.004 -4.953 -0.004 -4.953 -0.036 -0.014 -0.014  0.014  0.163  0.014  0.036 -0.014 -0.014  0.036]
 [ 0.111  0.021 -0.023  0.021 -0.564 -0.     0.004 -0.56  -0.    -0.564 -0.017 -0.002  0.    -0.002  0.009  0.     0.002 -0.002 -0.002 -0.017]
 [ 0.111  0.021  0.021 -0.023 -0.564  0.004 -0.    -0.564 -0.    -0.56  -0.017  0.    -0.002  0.     0.009 -0.002 -0.017 -0.002 -0.002  0.002]
 [ 0.111 -0.023  0.021  0.021 -0.56  -0.    -0.    -0.564  0.004 -0.564  0.002 -0.002 -0.002 -0.002  0.009 -0.002 -0.017  0.     0.    -0.017]
 [ 0.111 -0.021 -0.021 -0.023 -0.564  0.004  0.    -0.564  0.    -0.56   0.017 -0.    -0.002 -0.     0.009  0.002  0.017 -0.002  0.002  0.002]
 [ 0.111  0.023 -0.021  0.021 -0.56  -0.     0.    -0.564 -0.004 -0.564 -0.002  0.002 -0.002  0.002  0.009  0.002  0.017  0.    -0.    -0.017]
 [ 0.111 -0.021  0.023  0.021 -0.564 -0.    -0.004 -0.56   0.    -0.564  0.017  0.002  0.     0.002  0.009 -0.    -0.002 -0.002  0.002 -0.017]
 [ 0.111 -0.021 -0.023 -0.021 -0.564  0.     0.004 -0.56   0.    -0.564  0.017 -0.002 -0.     0.002  0.009 -0.     0.002  0.002 -0.002  0.017]
 [ 0.111 -0.021  0.021  0.023 -0.564 -0.004 -0.    -0.564  0.    -0.56   0.017  0.     0.002 -0.     0.009  0.002 -0.017  0.002 -0.002 -0.002]
 [ 0.111  0.023  0.021 -0.021 -0.56   0.    -0.    -0.564 -0.004 -0.564 -0.002 -0.002  0.002  0.002  0.009  0.002 -0.017 -0.     0.     0.017]
 [ 0.111  0.021 -0.021  0.023 -0.564 -0.004  0.    -0.564 -0.    -0.56  -0.017 -0.     0.002  0.     0.009 -0.002  0.017  0.002  0.002 -0.002]
 [ 0.111 -0.023 -0.021 -0.021 -0.56   0.     0.    -0.564  0.004 -0.564  0.002  0.002  0.002 -0.002  0.009 -0.002  0.017 -0.    -0.     0.017]
 [ 0.111  0.021  0.023 -0.021 -0.564  0.    -0.004 -0.56  -0.    -0.564 -0.017  0.002 -0.    -0.002  0.009  0.    -0.002  0.002  0.002  0.017]]
"""

# C6H6
# obj_func_type = 1: nb_iterations = 35
# obj_func_type = 2: nb_iterations = 44
"""
charges:
[-0.106 -0.107 -0.107 -0.107 -0.107 -0.106  0.106  0.106  0.107  0.107  0.106  0.106]
cartesian multipoles:
[[-0.106 -0.046  0.009 -0.    -4.246 -0.019 -0.    -4.338 -0.    -4.359 -0.268  0.113 -0.     0.119 -0.    -0.112 -0.082  0.     0.023 -0.   ]
 [-0.107 -0.015  0.045  0.    -4.335 -0.03   0.    -4.255 -0.    -4.361  0.12  -0.069  0.    -0.169 -0.    -0.037  0.216  0.     0.107 -0.   ]
 [-0.107 -0.032 -0.036 -0.    -4.303  0.049 -0.    -4.29  -0.    -4.362  0.08  -0.134 -0.    -0.183 -0.    -0.075  0.019 -0.    -0.085 -0.   ]
 [-0.107  0.032  0.036 -0.    -4.303  0.049 -0.    -4.29  -0.    -4.362 -0.08   0.134  0.     0.183 -0.     0.075 -0.019  0.     0.085  0.   ]
 [-0.107  0.015 -0.045  0.    -4.335 -0.03  -0.    -4.255  0.    -4.361 -0.12   0.069  0.     0.169 -0.     0.037 -0.216  0.    -0.107  0.   ]
 [-0.106  0.046 -0.009  0.    -4.246 -0.019  0.    -4.338  0.    -4.359  0.268 -0.113 -0.    -0.119 -0.     0.112  0.082  0.    -0.023  0.   ]
 [ 0.106  0.044 -0.009 -0.    -0.589  0.003 -0.    -0.576 -0.    -0.639 -0.003  0.004 -0.     0.009 -0.    -0.022 -0.006  0.     0.004 -0.   ]
 [ 0.106  0.014 -0.043 -0.    -0.577  0.005  0.    -0.589 -0.    -0.639  0.008 -0.006 -0.    -0.007 -0.    -0.007  0.     0.     0.022  0.   ]
 [ 0.107  0.03   0.034 -0.    -0.581 -0.007 -0.    -0.583 -0.    -0.639  0.01  -0.004 -0.    -0.006 -0.    -0.015  0.009 -0.    -0.017  0.   ]
 [ 0.107 -0.03  -0.034  0.    -0.581 -0.007  0.    -0.583 -0.    -0.639 -0.01   0.004  0.     0.006  0.     0.015 -0.009  0.     0.017  0.   ]
 [ 0.106 -0.014  0.043  0.    -0.577  0.005 -0.    -0.589  0.    -0.639 -0.008  0.006  0.     0.007 -0.     0.007 -0.     0.    -0.022  0.   ]
 [ 0.106 -0.044  0.009  0.    -0.589  0.003 -0.    -0.576  0.    -0.639  0.003 -0.004  0.    -0.009 -0.     0.022  0.006  0.    -0.004  0.   ]]
"""
