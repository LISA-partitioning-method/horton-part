import horton_grid
import grid
import numpy as np
import pytest
from .common import get_atoms_npz
from scipy.interpolate import CubicHermiteSpline


@pytest.mark.parametrize("fn", get_atoms_npz([1, 6], 1, -1, "pow", None))
def test_grid_pow(fn):
    with np.load(fn, mmap_mode=None) as npz:
        # number, charge, energy = (
        #     int(npz["number"]),
        #     int(npz["charge"]),
        #     float(npz["energy"]),
        # )
        charge = int(npz["charge"])
        dens, deriv = npz["dens"], npz["deriv"]

        rmin, rmax, npoint = npz["rgrid"]
        npoint = int(npoint)
        uniform_grid = grid.UniformInteger(npoint)

        rgrid = grid.PowerRTransform(rmin, rmax, npoint - 1).transform_1d_grid(
            uniform_grid
        )
        rtransform = horton_grid.PowerRTransform(rmin, rmax, npoint)
        rgrid2 = horton_grid.RadialGrid(rtransform)

        assert np.allclose(rgrid.points, rgrid2.radii)
        assert np.allclose(
            rgrid.weights * 4 * np.pi * rgrid.points**2, rgrid2.weights
        )
        assert rgrid.integrate(dens * 4 * np.pi * rgrid.points**2) == pytest.approx(
            rgrid2.integrate(dens), abs=1e-8
        )
        print(charge)

        print(rgrid.integrate(dens * 4 * np.pi * rgrid.points**2))

        spline = CubicHermiteSpline(rgrid.points, dens, deriv)
        spline2 = horton_grid.CubicSpline(dens, deriv, rtransform)

        assert spline(rgrid.points) == pytest.approx(spline2(rgrid.points), abs=1e-6)
        assert spline(rgrid.points) == pytest.approx(spline2.y, abs=1e-6)
        assert rgrid.points == pytest.approx(rtransform.get_radii(), abs=1e-6)
        assert spline.x == pytest.approx(rtransform.get_radii(), abs=1e-6)
