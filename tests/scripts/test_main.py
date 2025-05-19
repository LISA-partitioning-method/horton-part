# HORTON-PART: molecular density partition schemes based on HORTON package.
# Copyright (C) 2023-2025 The HORTON-PART Development Team
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
from importlib import resources

import numpy as np
import pytest

from horton_part.scripts.generate_density import main as main1
from horton_part.scripts.partition_density import main as main2


@pytest.mark.parametrize("type", [None, "lisa", "glisa", "mbis", "gisa", "is", "nlis", "gmbis"])
@pytest.mark.parametrize("fn_wfn", ["hf_sto3g.fchk", "water_sto3g_hf_g03.fchk"])
def test_part_dens_default_setup(fn_wfn, type, tmpdir):
    with resources.path("iodata.test.data", fn_wfn) as fn_full:
        fn_density = os.path.join(tmpdir, "density.npz")
        fn_out = os.path.join(tmpdir, "part.npz")
        fn_log = os.path.join(tmpdir, "density.log")

        yaml_file = tmpdir.join("input.yaml")
        if type is None:
            yaml_content = f"""
                part-gen:
                  inputs:
                  - {fn_full}
                  outputs:
                  - {fn_density}
                  log_files:
                  - {fn_log}
                part-dens:
                  inputs:
                  - {fn_density}
                  outputs:
                  - {fn_out}
                  log_files:
                  - {fn_log}
                """
        else:
            yaml_content = f"""
                part-gen:
                  inputs:
                  - {fn_full}
                  outputs:
                  - {fn_density}
                  log_files:
                  - {fn_log}
                part-dens:
                  inputs:
                  - {fn_density}
                  outputs:
                  - {fn_out}
                  log_files:
                  - {fn_log}
                  type : {type}
            """
        if type == "gisa":
            yaml_content = f"""
                part-gen:
                  inputs:
                  - {fn_full}
                  outputs:
                  - {fn_density}
                  log_files:
                  - {fn_log}
                part-dens:
                  inputs:
                  - {fn_density}
                  outputs:
                  - {fn_out}
                  log_files:
                  - {fn_log}
                  type : {type}
                  solver : 'quadprog'
            """

        yaml_file.write(yaml_content)

        main1([str(yaml_file)])
        main2([str(yaml_file)])
        assert os.path.isfile(fn_density)
        assert os.path.isfile(fn_out)
        data = dict(np.load(fn_out))

    keys = [
        "history_entropies",
        "history_charges",
        "history_propars",
        "time",
        "time_update_at_weights",
        "time_update_propars",
        "niter",
        "charges",
        "natom",
        "atnums",
        "atcorenums",
        "lmax",
        "maxiter",
        "threshold",
    ]
    for k in keys:
        assert k in data
