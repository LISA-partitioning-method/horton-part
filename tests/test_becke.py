#!/usr/bin/env python
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


from grid import BeckeWeights, ExpRTransform, MolGrid, UniformInteger
from pytest import raises

from horton_part.becke import BeckeWPart

from .common import load_molecule_npz, reorder_rows


def test_becke_n2_hfs_sto3g():
    # load molecule data
    coords, nums, pseudo_nums, dens, points = load_molecule_npz(
        "n2_hfs_sto3g_fchk_exp:1e-3:1e1:100:110.npz"
    )
    # make grid
    rtf = ExpRTransform(1e-3, 1e1, 100 - 1)
    uniform_grid = UniformInteger(100)
    rgrid = rtf.transform_1d_grid(uniform_grid)
    becke = BeckeWeights()
    grid = MolGrid.from_size(nums, coords, rgrid, 110, becke, rotate=False, store=True)
    # check the grid points against stored points on which density is evaluated
    points_sorted, new_sort = reorder_rows(grid.points, points, return_order=True)
    dens_sorted = dens[new_sort]
    assert (abs(points_sorted - grid.points) < 1.0e-6).all()
    # Becke partitioning
    bp = BeckeWPart(coords, nums, pseudo_nums, grid, dens_sorted)
    bp.do_populations()
    assert abs(bp["populations"] - 7).max() < 1e-4
    bp.do_charges()
    assert abs(bp["charges"]).max() < 1e-4
    bp.clear()
    with raises(KeyError):
        bp["charges"]
    bp.do_charges()
    assert abs(bp["populations"] - 7).max() < 1e-4
    assert abs(bp["charges"]).max() < 1e-4
