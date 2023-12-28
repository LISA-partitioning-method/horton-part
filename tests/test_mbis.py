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


from horton_part.mbis import _get_initial_mbis_propars, _get_nshell


def test_get_nshell():
    assert _get_nshell(1) == 1
    assert _get_nshell(2) == 1
    assert _get_nshell(3) == 2


def test_get_initial_mbis_propars():
    assert (_get_initial_mbis_propars(1) == [1.0, 2.0]).all()
    assert (_get_initial_mbis_propars(2) == [2.0, 4.0]).all()
    assert (_get_initial_mbis_propars(3) == [2.0, 6.0, 1.0, 2.0]).all()
