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

import numpy as np
import pytest

from horton_part.scripts.generate_density import main
from horton_part.scripts.partition_density import construct_molgrid_from_dict


@pytest.mark.parametrize("fn_wfn", ["hf_sto3g.fchk", "water_sto3g_hf_g03.fchk"])
def test_construct_molgrid_from_dict(fn_wfn, tmpdir):
    with resources.path("iodata.test.data", fn_wfn) as fn_full:
        fn_density = os.path.join(tmpdir, "density.npz")
        fn_log = os.path.join(tmpdir, "density.log")

        yaml_file = tmpdir.join("input.yaml")
        yaml_content = f"""
            part-gen:
              inputs:
              - {fn_full}
              outputs:
              - {fn_density}
              log_files:
              - {fn_log}
            """
        yaml_file.write(yaml_content)

        with pytest.warns(None):
            main([str(yaml_file)])
        assert os.path.isfile(fn_density)
        data = dict(np.load(fn_density))
        molgrid = construct_molgrid_from_dict(data)

    natom = len(data["atnums"])
    for iatom in range(natom):
        atgrid = molgrid.get_atomic_grid(iatom)
        assert atgrid.points.shape == data[f"atom{iatom}/points"].shape
        assert atgrid.weights.shape == data[f"atom{iatom}/weights"].shape
        assert atgrid.points == pytest.approx(data[f"atom{iatom}/points"], abs=1e-8)
        assert atgrid.weights == pytest.approx(data[f"atom{iatom}/weights"], abs=1e-8)

    assert molgrid.points.shape == data["points"].shape
    assert molgrid.weights.shape == data["weights"].shape
    assert molgrid.points == pytest.approx(data["points"], abs=1e-8)
    assert molgrid.weights == pytest.approx(data["weights"], abs=1e-8)
