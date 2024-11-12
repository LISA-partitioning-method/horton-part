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
import pytest

from horton_part.nlis import get_initial_nlis_propars, get_nlis_nshell


def load_example_nshell_dict(nshell=4):
    """Load an example of `nshell_dict`."""
    return {n: nshell for n in range(1, 119)}


def load_example_exp_n_dict(number, nshell):
    """Load an example of `exp_n_dict`."""
    exp_n_dict = {(number, ishell): 2 for ishell in range(nshell)}
    return exp_n_dict


@pytest.mark.parametrize("nshell", [4, 5, 6])
def test_get_nlis_nshell(nshell):
    nshell_dict = load_example_nshell_dict(nshell)
    assert get_nlis_nshell(1, nshell_dict) == nshell
    assert get_nlis_nshell(2, nshell_dict) == nshell
    assert get_nlis_nshell(3, nshell_dict) == nshell
    assert get_nlis_nshell(17, nshell_dict) == nshell
    assert get_nlis_nshell(18, nshell_dict) == nshell
    assert get_nlis_nshell(21, nshell_dict) == nshell
    assert get_nlis_nshell(44, nshell_dict) == nshell
    assert get_nlis_nshell(72, nshell_dict) == nshell
    assert get_nlis_nshell(104, nshell_dict) == nshell


@pytest.mark.parametrize("nshell", [4])
def test_get_initial_nlis_propars(nshell):
    nshell_dict = load_example_nshell_dict(nshell)
    values_h = get_initial_nlis_propars(1, load_example_exp_n_dict(1, nshell), nshell_dict)
    assert values_h == pytest.approx(
        [
            1 / nshell,
            2.0,
            2.0,
            1 / nshell,
            1.25992105,
            2.0,
            1 / nshell,
            0.79370053,
            2.0,
            1 / nshell,
            0.5,
            2.0,
        ]
    )
