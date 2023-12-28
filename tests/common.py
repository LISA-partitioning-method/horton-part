# HORTON-PART: molecular density partition schemes based on HORTON package.
# Copyright (C) 2023-2024 The HORTON-PART Development Team
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


import os
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
import pytest
from grid import PowerRTransform, UniformInteger

from horton_part.core.proatomdb import ProAtomRecord

__all__ = [
    "get_fn",
    "load_molecule_npz",
    "load_atoms_npz",
    "check_names",
    "check_proatom_splines",
    "reorder_rows",
]


def get_fn(fn):
    cur_pth = os.path.split(__file__)[0]
    return f"{cur_pth}/cached/{fn}"


def load_molecule_npz(filename, spin_dens=False):
    # get file path
    filepath = get_fn(filename)
    # load npz file
    with np.load(filepath, mmap_mode=None) as npz:
        dens = npz["dens"]
        points = npz["points"]
        numbers = npz["numbers"]
        coordinates = npz["coordinates"]
        pseudo_numbers = npz["pseudo_numbers"]
        if spin_dens:
            spindens = npz["spin_dens"]
            return coordinates, numbers, pseudo_numbers, dens, spindens, points
    return coordinates, numbers, pseudo_numbers, dens, points


def get_atoms_npz(numbers, max_cation, max_anion, rtf_type, level):
    filepaths = []
    for number in numbers:
        nelectrons = number - np.arange(max_anion, max_cation + 1)
        for nelec in nelectrons:
            if level is None:
                filename = get_fn("atom_Z%.2i_N%.2i_%s.npz" % (number, nelec, rtf_type))
            else:
                filename = get_fn("atom_%s_Z%.2i_N%.2i_%s.npz" % (level, number, nelec, rtf_type))
            if os.path.isfile(filename):
                filepaths.append(filename)
    return filepaths


def load_atoms_npz(numbers, max_cation, max_anion, rtf_type="pow", level=None):
    # radial transformation available
    rtf_classes = {
        # "lin": LinearFiniteRTransform,
        # "exp": ExpRTransform,
        "pow": PowerRTransform,
    }
    # get filepath of atoms npz
    filepaths = get_atoms_npz(numbers, max_cation, max_anion, rtf_type, level)
    # load each file into a record
    records = []
    rtfclass = rtf_classes[rtf_type]
    for filepath in filepaths:
        with np.load(filepath, mmap_mode=None) as npz:
            number, charge, energy = (
                int(npz["number"]),
                int(npz["charge"]),
                float(npz["energy"]),
            )
            dens, deriv = npz["dens"], npz["deriv"]
            rmin, rmax, npoint = npz["rgrid"]
            npoint = int(npoint)
            uniform_grid = UniformInteger(npoint)
            rgrid = rtfclass(rmin, rmax, npoint - 1).transform_1d_grid(uniform_grid)
            if "pseudo_number" in list(npz.keys()):
                pseudo_number = npz["pseudo_number"]
            else:
                pseudo_number = None
            record = ProAtomRecord(
                number, charge, energy, rgrid, dens, deriv, pseudo_number=pseudo_number
            )
            records.append(record)
    return records


@contextmanager
def tmpdir(name):
    dn = tempfile.mkdtemp(name)
    try:
        yield dn
    finally:
        shutil.rmtree(dn)


def check_names(names, part):
    for name in names:
        assert name in part.cache


def check_proatom_splines(part):
    for index in range(part.natom):
        spline = part.get_proatom_spline(index)
        grid = part.get_grid(index)
        # array1 = grid.zeros()
        array1 = np.zeros_like(grid.weights)
        part.eval_spline(index, spline, array1, grid)
        # array2 = grid.zeros()
        array2 = np.zeros_like(grid.weights)
        part.eval_proatom(index, array2, grid)
        assert abs(array1).max() != 0.0
        assert abs(array1 - array2).max() < 1e-5


def reorder_rows(A, B, return_order=False, decimals=8):
    assert A.shape == B.shape
    if A.size == 0 or B.size == 0:
        raise ValueError("Input matrices must not be empty")
    A_tuples = [tuple(np.round(row, decimals)) for row in A]
    B_tuples = [tuple(np.round(row, decimals)) for row in B]
    index_map_B = {row: i for i, row in enumerate(B_tuples)}
    new_order = [index_map_B[row] for row in A_tuples]
    B_sorted = B[new_order]
    if return_order:
        return B_sorted, new_order
    else:
        return B_sorted


def test_reorder_rows():
    A = np.array([[3, 2, 1], [1, 2, 3], [2, 3, 1]])
    B = np.array([[1, 2, 3], [2, 3, 1], [3, 2, 1]])
    B_sorted = reorder_rows(A, B)
    expected_output = np.array([[3, 2, 1], [1, 2, 3], [2, 3, 1]])
    np.testing.assert_array_equal(B_sorted, expected_output)


def test_reorder_rows_same_matrix():
    A = np.array([[1, 2, 3], [2, 3, 1], [3, 2, 1]])
    B = np.array([[1, 2, 3], [2, 3, 1], [3, 2, 1]])
    B_sorted = reorder_rows(A, B)
    expected_output = np.array([[1, 2, 3], [2, 3, 1], [3, 2, 1]])
    np.testing.assert_array_equal(B_sorted, expected_output)


def test_reorder_rows_reverse_order():
    A = np.array([[3, 2, 1], [2, 3, 1], [1, 2, 3]])
    B = np.array([[1, 2, 3], [2, 3, 1], [3, 2, 1]])
    B_sorted = reorder_rows(A, B)
    expected_output = np.array([[3, 2, 1], [2, 3, 1], [1, 2, 3]])
    np.testing.assert_array_equal(B_sorted, expected_output)


def test_reorder_rows_empty_matrices():
    A = np.array([[]])
    B = np.array([[]])
    with pytest.raises(ValueError):
        reorder_rows(A, B)


def test_reorder_rows_different_shape():
    A = np.array([[3, 2, 1], [1, 2, 3]])
    B = np.array([[1, 2, 3], [2, 3, 1], [3, 2, 1]])
    with pytest.raises(AssertionError):
        reorder_rows(A, B)
