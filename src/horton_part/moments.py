# -*- coding: utf-8 -*-
# HORTON-PART: GRID for Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2023 The HORTON-PART Development Team
#
# This file is part of HORTON-PART
#
# HORTON-PART is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON-PART is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --
from copy import deepcopy
import numpy as np
from gbasis.evals.eval import evaluate_basis

__all__ = [
    "get_npure_cumul",
    "fill_pure_polynomials",
    "compute_aim_transition_multipole_moment",
]


def get_npure_cumul(lmax):
    """The number of pure functions up to a given angular momentum, lmax."""
    return (lmax + 1) ** 2


def fill_pure_polynomials(output, lmax):
    # Shell l=0
    if lmax <= 0:
        return -1

    # Shell l=1
    if lmax <= 1:
        return 0

    # Shell l>1

    # auxiliary variables
    z = output[:, 0]
    x = output[:, 1]
    y = output[:, 2]
    r2 = x * x + y * y + z * z

    # work arrays in which the PI(z,r) polynomials are stored.
    npoint = output.shape[0]
    pi_old = np.zeros((npoint, lmax + 1))
    pi_new = np.zeros((npoint, lmax + 1))
    a = np.zeros((npoint, lmax + 1))
    b = np.zeros((npoint, lmax + 1))

    # initialize work arrays
    pi_old[:, 0] = 1
    pi_new[:, 0] = z
    pi_new[:, 1] = 1
    a[:, 1] = x
    b[:, 1] = y

    old_offset = 0  # first array index of the moments of the previous shell
    old_npure = 3  # number of moments in previous shell

    for l in range(2, lmax + 1):
        # numbers for current iteration
        new_npure = old_npure + 2
        new_offset = old_offset + old_npure

        # construct polynomials PI(z,r) for current l
        factor = 2 * l - 1
        for m in range(l - 1):
            tmp = deepcopy(pi_old[:, m])
            pi_old[:, m] = pi_new[:, m]
            pi_new[:, m] = (z * factor * pi_old[:, m] - r2 * (l + m - 1) * tmp) / (
                l - m
            )

        pi_old[:, l - 1] = pi_new[:, l - 1]
        pi_new[:, l] = factor * pi_old[:, l - 1]
        pi_new[:, l - 1] = z * pi_new[:, l]

        # construct new polynomials A(x,y) and B(x,y)
        a[:, l] = x * a[:, l - 1] - y * b[:, l - 1]
        b[:, l] = y * a[:, l - 1] + x * b[:, l - 1]

        # construct solid harmonics
        output[:, new_offset] = pi_new[:, 0]
        factor = np.sqrt(2)
        for m in range(1, l + 1):
            factor /= np.sqrt((l + m) * (l - m + 1))
            output[:, new_offset + 2 * m - 1] = factor * a[:, m] * pi_new[:, m]
            output[:, new_offset + 2 * m] = factor * b[:, m] * pi_new[:, m]

        # translate new to old numbers
        old_npure = new_npure
        old_offset = new_offset

    return old_offset


def compute_aim_transition_multipole_moment(
    molgrid, basis, coord_type, coordinates, aim_weights, lmax, overlap=None
):
    r"""Compute Fock matrix of potential function based on AIM function.

    .. math::

        \langle \phi_i | \omega_a(r) R_k(r) | \phi_j \rangle

    where :math:`\phi_i` is orbital `i`, :math:`\omega_a(r)` is weight function of
    atom `a` and :math:`R_k(r)` is real spherical harmonics.

    """
    natom = coordinates.shape[0]
    # compute overlap operators
    npure = get_npure_cumul(lmax)

    overlap_operators = {}
    for iatom in range(natom):
        # Prepare solid harmonics on atgrids.
        atgrid = molgrid.atgrids[iatom]
        if lmax > 0:
            work = np.zeros((atgrid.size, npure - 1), float)
            work[:, 0] = atgrid.points[:, 2] - coordinates[iatom, 2]  # z
            work[:, 1] = atgrid.points[:, 0] - coordinates[iatom, 0]  # x
            work[:, 2] = atgrid.points[:, 1] - coordinates[iatom, 1]  # y
            if lmax > 1:
                fill_pure_polynomials(work, lmax)
        else:
            work = None

        # Convert the weight functions to AIM overlap operators.
        aim_weight = aim_weights[iatom]
        for ipure in range(npure):
            if ipure > 0:
                moment = aim_weight * work[:, ipure - 1]
            else:
                moment = aim_weight
            # convert weight functions to matrix based on basis sets
            aobasis = evaluate_basis(
                basis, atgrid.points, transform=None, coord_type=coord_type
            )
            op = np.einsum("mi,ni,i,i->mn", aobasis, aobasis, moment, atgrid.weights)
            # op = compute_grid_density_fock(moment, atgrid.points, atgrid.weights)
            overlap_operators[(iatom, ipure)] = op

    if overlap is not None:
        # Correct the s-type overlap operators such that the sum is exactly equal to the total
        # overlap.
        calc_olp = 0.0
        for i in range(natom):
            calc_olp += overlap_operators[(i, 0)]
        error_olp = (calc_olp - overlap) / natom
        for i in range(natom):
            overlap_operators[(i, 0)] - error_olp

    # sort the operators
    result = []
    # sort the response function basis
    for ipure in range(npure):
        for iatom in range(natom):
            result.append(overlap_operators[(iatom, ipure)])
    return np.asarray(result)
