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
"""Test the input preparation with HORTON3 modules."""

import os
from importlib import resources

import pytest

from horton_part.scripts.generate_cube import main

FILENAMES = [
    "water_hfs_321g.fchk",
]


@pytest.mark.parametrize("fn_wfn", FILENAMES)
def test_from_horton3_density(fn_wfn, tmpdir):
    with resources.path("iodata.test.data", fn_wfn) as fn_full:
        fn_density = os.path.join(tmpdir, "density.npz")
        fn_log = os.path.join(tmpdir, "density.log")

        yaml_file = tmpdir.join("input.yaml")
        yaml_content = f"""
            part-cube:
              inputs:
              - {fn_full}
              outputs:
              - {fn_density}
              log_files:
              - {fn_log}
              chunk_size : 1000
              gradient : false
              orbitals : false
              log_level: INFO
            """
        yaml_file.write(yaml_content)

        # Run the main function using the YAML file as input
        with pytest.warns(None):
            main([str(yaml_file)])
        assert os.path.isfile(fn_density)
        # data = dict(np.load(fn_density))
        # fn_name = ".".join(os.path.basename(fn_wfn).split(".")[:-1])
        # fn_cube = os.path.join(tmpdir, f"{fn_name}_rho_mol.cube")
        # print(fn_cube)
        # assert os.path.isfile(fn_cube)
