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

from horton_part.scripts.generate_density import main as main_gen
from horton_part.scripts.partition_density import construct_molgrid_from_dict
from horton_part.scripts.partition_density import main as main_part


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

        main_gen([str(yaml_file)])
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


@pytest.mark.parametrize("part_type", ["lisa", "glisa", "mbis", "gisa", "is", "nlis", "gmbis"])
@pytest.mark.parametrize("fn_wfn", ["water_sto3g_hf_g03.fchk"])
def test_part_dens(part_type, fn_wfn, tmpdir):
    with resources.path("iodata.test.data", fn_wfn) as fn_full:
        fn_density = os.path.join(tmpdir, "density.npz")
        fn_log = os.path.join(tmpdir, "density.log")

        yaml_file_gen = tmpdir.join("input_gen.yaml")
        yaml_content_gen = f"""
            part-gen:
              inputs:
              - {fn_full}
              outputs:
              - {fn_density}
              log_files:
              - {fn_log}
            """
        yaml_file_gen.write(yaml_content_gen)

        main_gen([str(yaml_file_gen)])
        assert os.path.isfile(fn_density)

        fn_part = os.path.join(tmpdir, "part.npz")
        yaml_file_part = tmpdir.join("input_part.yaml")
        yaml_content_part = f"""
            part-dens:
              inputs:
              - {fn_density}
              outputs:
              - {fn_part}
              log_files:
              - {fn_log}
              type : {part_type}
              solver: {"sc" if part_type != 'gisa' else 'quadprog'}
        """
        yaml_file_part.write(yaml_content_part)

        main_part([str(yaml_file_part)])
        assert os.path.isfile(fn_part)


@pytest.mark.slow
@pytest.mark.parametrize("solver", ["sc", "cvxopt", "sc-1-iter", "sc-plus-convex"])
@pytest.mark.parametrize("fn_wfn", ["water_sto3g_hf_g03.fchk"])
def test_lisa_with_mol_grids(solver, fn_wfn, tmpdir):
    with resources.path("iodata.test.data", fn_wfn) as fn_full:
        fn_density = os.path.join(tmpdir, "density.npz")
        fn_log = os.path.join(tmpdir, "density.log")

        yaml_file_gen = tmpdir.join("input_gen.yaml")
        yaml_content_gen = f"""
            part-gen:
              inputs:
              - {fn_full}
              outputs:
              - {fn_density}
              log_files:
              - {fn_log}
            """
        yaml_file_gen.write(yaml_content_gen)

        main_gen([str(yaml_file_gen)])
        assert os.path.isfile(fn_density)

        fn_part = os.path.join(tmpdir, "part.npz")
        yaml_file_part = tmpdir.join("input_part.yaml")
        yaml_content_part = f"""
            part-dens:
              inputs:
              - {fn_density}
              outputs:
              - {fn_part}
              log_files:
              - {fn_log}
              type : lisa
              solver : {solver}
              grid_type : 2
        """
        yaml_file_part.write(yaml_content_part)

        main_part([str(yaml_file_part)])
        assert os.path.isfile(fn_part)


@pytest.mark.parametrize("grid_type", [1, 2, 3])
@pytest.mark.parametrize("solver", ["sc", "cvxopt"])
@pytest.mark.parametrize("method", ["glisa", "lisa"])
@pytest.mark.parametrize("fn_wfn", ["water_sto3g_hf_g03.fchk"])
def test_glisa_with_molgrid(grid_type, solver, method, fn_wfn, tmpdir):
    with resources.path("iodata.test.data", fn_wfn) as fn_full:
        fn_density = os.path.join(tmpdir, "density.npz")
        fn_log = os.path.join(tmpdir, "density.log")

        yaml_file_gen = tmpdir.join("input_gen.yaml")
        yaml_content_gen = f"""
            part-gen:
              inputs:
              - {fn_full}
              outputs:
              - {fn_density}
              log_files:
              - {fn_log}
            """
        yaml_file_gen.write(yaml_content_gen)

        main_gen([str(yaml_file_gen)])
        assert os.path.isfile(fn_density)

        fn_part = os.path.join(tmpdir, "part.npz")
        yaml_file_part = tmpdir.join("input_part.yaml")
        yaml_content_part = f"""
            part-dens:
              inputs:
              - {fn_density}
              outputs:
              - {fn_part}
              log_files:
              - {fn_log}
              type : {method}
              solver : {solver}
              grid_type : {grid_type}
        """
        yaml_file_part.write(yaml_content_part)

        main_part([str(yaml_file_part)])
        assert os.path.isfile(fn_part)
        data = np.load(fn_part)
        assert data["charges"].size == 3
        assert np.sum(data["charges"]) == pytest.approx(0.00, abs=1e-4)
        assert data["charges"][0] < 0 and data["charges"][1] > 0 and data["charges"][2] > 0


@pytest.mark.parametrize("grid_type", [1, 2, 3])
@pytest.mark.parametrize("solver", ["sc"])
@pytest.mark.parametrize("method", ["gmbis"])
@pytest.mark.parametrize("fn_wfn", ["water_sto3g_hf_g03.fchk"])
def test_gmbis(grid_type, solver, method, fn_wfn, tmpdir):
    with resources.path("iodata.test.data", fn_wfn) as fn_full:
        fn_density = os.path.join(tmpdir, "density.npz")
        fn_log = os.path.join(tmpdir, "density.log")

        yaml_file_gen = tmpdir.join("input_gen.yaml")
        yaml_content_gen = f"""
            part-gen:
              inputs:
              - {fn_full}
              outputs:
              - {fn_density}
              log_files:
              - {fn_log}
            """
        yaml_file_gen.write(yaml_content_gen)

        main_gen([str(yaml_file_gen)])
        assert os.path.isfile(fn_density)

        fn_part = os.path.join(tmpdir, "part.npz")
        yaml_file_part = tmpdir.join("input_part.yaml")
        yaml_content_part = f"""
            part-dens:
              inputs:
              - {fn_density}
              outputs:
              - {fn_part}
              log_files:
              - {fn_log}
              type : {method}
              solver : {solver}
              grid_type : {grid_type}
        """
        yaml_file_part.write(yaml_content_part)

        main_part([str(yaml_file_part)])
        assert os.path.isfile(fn_part)
        data = np.load(fn_part)
        assert data["charges"].size == 3
        abs = 1e-3 if method in ["mbis", "nlis", "gmbis"] else 1e-4
        assert np.sum(data["charges"]) == pytest.approx(0.00, abs=abs)
        assert data["charges"][0] < 0 and data["charges"][1] > 0 and data["charges"][2] > 0
        assert data["charges"] == pytest.approx(
            [-0.6179280883502, 0.30913844981854, 0.30918595758548], abs=1e-2
        )


@pytest.mark.parametrize("grid_type", [1, 2, 3])
@pytest.mark.parametrize("solver", ["sc"])
@pytest.mark.parametrize("method", ["mbis"])
@pytest.mark.parametrize("fn_wfn", ["water_sto3g_hf_g03.fchk"])
def test_mbis(grid_type, solver, method, fn_wfn, tmpdir):
    with resources.path("iodata.test.data", fn_wfn) as fn_full:
        fn_density = os.path.join(tmpdir, "density.npz")
        fn_log = os.path.join(tmpdir, "density.log")

        yaml_file_gen = tmpdir.join("input_gen.yaml")
        yaml_content_gen = f"""
            part-gen:
              inputs:
              - {fn_full}
              outputs:
              - {fn_density}
              log_files:
              - {fn_log}
            """
        yaml_file_gen.write(yaml_content_gen)

        main_gen([str(yaml_file_gen)])
        assert os.path.isfile(fn_density)

        fn_part = os.path.join(tmpdir, "part.npz")
        yaml_file_part = tmpdir.join("input_part.yaml")
        yaml_content_part = f"""
            part-dens:
              inputs:
              - {fn_density}
              outputs:
              - {fn_part}
              log_files:
              - {fn_log}
              type : {method}
              solver : {solver}
              grid_type : {grid_type}
        """
        yaml_file_part.write(yaml_content_part)

        main_part([str(yaml_file_part)])
        assert os.path.isfile(fn_part)
        data = np.load(fn_part)
        assert data["charges"].size == 3
        abs = 1e-3 if method in ["mbis", "nlis", "gmbis"] else 1e-4
        assert np.sum(data["charges"]) == pytest.approx(0.00, abs=abs)
        assert data["charges"][0] < 0 and data["charges"][1] > 0 and data["charges"][2] > 0
        assert data["charges"] == pytest.approx(
            [-0.6179280883502, 0.30913844981854, 0.30918595758548], abs=1e-2
        )


@pytest.mark.parametrize("grid_type", [1, 2, 3])
@pytest.mark.parametrize("solver", ["sc"])
@pytest.mark.parametrize("method", ["nlis"])
@pytest.mark.parametrize("fn_wfn", ["water_sto3g_hf_g03.fchk"])
def test_nlis(grid_type, solver, method, fn_wfn, tmpdir):
    with resources.path("iodata.test.data", fn_wfn) as fn_full:
        fn_density = os.path.join(tmpdir, "density.npz")
        fn_log = os.path.join(tmpdir, "density.log")

        yaml_file_gen = tmpdir.join("input_gen.yaml")
        yaml_content_gen = f"""
            part-gen:
              inputs:
              - {fn_full}
              outputs:
              - {fn_density}
              log_files:
              - {fn_log}
            """
        yaml_file_gen.write(yaml_content_gen)

        main_gen([str(yaml_file_gen)])
        assert os.path.isfile(fn_density)

        fn_part = os.path.join(tmpdir, "part.npz")
        yaml_file_part = tmpdir.join("input_part.yaml")
        yaml_content_part = f"""
            part-dens:
              inputs:
              - {fn_density}
              outputs:
              - {fn_part}
              log_files:
              - {fn_log}
              type : {method}
              solver : {solver}
              grid_type : {grid_type}
        """
        yaml_file_part.write(yaml_content_part)

        main_part([str(yaml_file_part)])
        assert os.path.isfile(fn_part)
        data = np.load(fn_part)
        assert data["charges"].size == 3
        abs = 1e-3 if method in ["mbis", "nlis", "gmbis"] else 1e-4
        assert np.sum(data["charges"]) == pytest.approx(0.00, abs=abs)
        print(data["charges"])
        assert data["charges"][0] < 0 and data["charges"][1] > 0 and data["charges"][2] > 0
        assert data["charges"] == pytest.approx(
            [-0.6179280883502, 0.30913844981854, 0.30918595758548], abs=1e-2
        )
