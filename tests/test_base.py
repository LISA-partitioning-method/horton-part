#!/usr/bin/env python
# -*- coding: utf-8 -*-
# HORTON: Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2017 The HORTON Development Team
#
# This file is part of HORTON.
#
# HORTON is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 3
# of the License, or (at your option) any later version.
#
# HORTON is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>
#
# --


from nose.tools import assert_raises

from .common import load_molecule_npz, reorder_rows
from horton_part.base import WPart

from grid import ExpRTransform, UniformInteger, BeckeWeights, MolGrid


def test_base_exceptions():
    # load molecule data
    coords, nums, pseudo_nums, dens, points = load_molecule_npz(
        "n2_hfs_sto3g_fchk_exp:1e-3:1e1:100:110.npz"
    )
    # make grid
    uniform_grid = UniformInteger(100)
    rtf = ExpRTransform(1e-3, 1e1, 100 - 1)
    rgrid = rtf.transform_1d_grid(uniform_grid)
    becke = BeckeWeights()
    grid = MolGrid.from_size(nums, coords, rgrid, 110, becke, rotate=False)
    # check the grid points against stored points on which density is evaluated
    points_reordered, order = reorder_rows(points, grid.points, return_order=True)
    dens = dens[order]
    assert (abs(points - points_reordered) < 1.0e-6).all()

    with assert_raises(ValueError):
        # the default setting is local=true, which is not compatible with store=False.
        WPart(coords, nums, pseudo_nums, grid, dens)

    grid = MolGrid.from_size(nums, coords, rgrid, 110, becke, rotate=False, store=True)
    with assert_raises(NotImplementedError):
        # It should not be possible to create instances of the base class.
        WPart(coords, nums, pseudo_nums, grid, dens)
