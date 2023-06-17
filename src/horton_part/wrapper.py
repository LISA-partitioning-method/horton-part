from scipy.interpolate import CubicHermiteSpline

import horton_grid
import grid
import numpy as np

__all__ = [
    "AtomicGrid",
    # "becke_helper_atom",
    "RTransform",
    "AtomicGridSpec",
    "CubicSpline",
    "ExpRTransform",
    "LinearRTransform",
    "PowerRTransform",
    "RadialGrid",
    "BeckeMolGrid",
    "solve_poisson_becke",
    "log",
    "biblio",
]


class CubicSpline(CubicHermiteSpline):
    def __init__(self, y, dx, rtransform, axis=0, extrapolate=None):
        self.y = y
        self.dx = dx
        self.rtransform = rtransform
        super().__init__(rtransform.get_radii(), y, dx, axis, extrapolate)


class RTransform(object):
    def __init__(self, npoint: int):
        self.npoint = int(npoint)
        # uniform grid
        self.x = np.arange(npoint, dtype=float)

    def get_deriv(self):
        return self.deriv(self.x)

    def get_radii(self):
        return self.transform(self.x)

    def to_string(self):
        return " ".join(
            [
                self.__class__.__name__,
                repr(self.rmin),
                repr(self.rmax),
                repr(self.npoint),
            ]
        )

    def chop(self, npoint):
        rmax = self.get_radii()[npoint - 1]
        return self.__class__(self.rmin, rmax, npoint)

    def radius(self, x):
        return self.transform(x)


class PowerRTransform(RTransform, grid.PowerRTransform):
    def __init__(self, rmin: float, rmax: float, npoint: int):
        RTransform.__init__(self, npoint)
        grid.PowerRTransform.__init__(self, rmin, rmax, b=npoint - 1)


class ExpRTransform(RTransform, grid.ExpRTransform):
    def __init__(self, rmin: float, rmax: float, npoint: int):
        RTransform.__init__(self, npoint)
        grid.ExpRTransform.__init__(self, rmin, rmax, b=npoint - 1)


class LinearRTransform(RTransform, grid.LinearInfiniteRTransform):
    def __init__(self, rmin: float, rmax: float, npoint: int):
        RTransform.__init__(self, npoint)
        grid.LinearInfiniteRTransform.__init__(self, rmin, rmax, b=npoint - 1)


class RadialGrid(grid.OneDGrid):
    def __init__(self, rtransform):
        self._rtransform = rtransform
        _grid = rtransform.transform_1d_grid(grid.UniformInteger(rtransform.npoint))
        super().__init__(
            _grid.points, 4 * np.pi * _grid.points**2 * _grid.weights, _grid.domain
        )

    def __eq__(self, other):
        return self.rtransform.to_string() == other.rtransform.to_string()

    def __ne__(self, other):
        return not self.__eq__(other)

    def _get_size(self):
        """The size of the grid."""
        return self._weights.size

    size = property(_get_size)

    def _get_shape(self):
        """The shape of the grid."""
        return self._weights.shape

    shape = property(_get_shape)

    def _get_rtransform(self):
        """The RTransform object of the grid."""
        return self._rtransform

    rtransform = property(_get_rtransform)

    def _get_weights(self):
        """The grid weights."""
        return self._weights

    weights = property(_get_weights)

    def _get_radii(self):
        """The positions of the radial grid points."""
        return self._points

    radii = property(_get_radii)

    def zeros(self):
        return np.zeros(self.shape)

    def chop(self, new_size):
        """Return a radial grid with a different number of points.

         **Arguments:**

         new_size
             The new number of radii.

        The corresponding radii remain the same.
        """
        rtf = self._rtransform.chop(new_size)
        return RadialGrid(rtf)


AtomicGrid = horton_grid.AtomicGrid
# becke_helper_atom = becke_helper_atom
# RTransform = RTransform
# CubicSpline = CubicSpline
# ExpRTransform = ExpRTransform
# LinearRTransform = LinearRTransform
# PowerRTransform = PowerRTransform
# RadialGrid = RadialGrid
# BeckeMolGrid = grid.MolGrid
AtomicGridSpec = horton_grid.AtomicGridSpec
BeckeMolGrid = horton_grid.BeckeMolGrid
solve_poisson_becke = horton_grid.solve_poisson_becke
log = horton_grid.log
biblio = horton_grid.biblio
