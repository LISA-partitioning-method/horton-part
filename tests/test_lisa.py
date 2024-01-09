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

#!/usr/bin/env python

import logging

import numpy as np
import pytest
from grid.onedgrid import GaussChebyshev
from grid.rtransform import BeckeRTransform

from horton_part.lisa import (
    opt_propars_convex_opt,
    opt_propars_diis,
    opt_propars_self_consistent,
    opt_propars_trust_region,
)

logger = logging.getLogger(__name__)


class GaussianFunction:
    def __init__(self, c_ak, alpha):
        self.population, self.exponent = c_ak, alpha

    def compute(self, points, nderiv=0):
        """Evaluate function values on points."""
        f = self.population * (self.exponent / np.pi) ** 1.5 * np.exp(-self.exponent * points**2)
        if nderiv == 0:
            return f
        elif nderiv == 1:
            return -2 * self.exponent * points * f
        else:
            raise NotImplementedError

    def get_cutoff_radius(self, density_cutoff):
        """Compute cutoff radius based on `density_cutoff`."""
        if density_cutoff <= 0.0:
            return np.inf
        prefactor = self.population * (self.exponent / np.pi) ** 1.5
        # if prefactor < 0 or prefactor < density_cutoff:
        if prefactor < 0:
            prefactor = -prefactor
        if prefactor < density_cutoff:
            return np.inf
        else:
            return np.sqrt((np.log(prefactor) - np.log(density_cutoff)) / self.exponent)


def _setup_rgrid(nrad=150):
    return BeckeRTransform(1e-4, 1.5).transform_1d_grid(GaussChebyshev(nrad))


def check_rho(rho):
    """Check whether a density is positive anywhere and monotonically decaying."""
    assert np.all(rho >= 0) and np.all(rho[:-1] >= rho[1:])


@pytest.mark.parametrize(
    "case",
    [
        ([0.3, 0.3, 0.510, -0.11], [10.0, 100.0, 1.0, 50]),
        ([0.2, 0.3, 0.500], [10.0, 20.0, 1.0]),
        ([0.2, 0.3, 0.250, 0.25], [10.0, 20.0, 1.0, 15]),
    ],
)
@pytest.mark.parametrize(
    "opt_propars",
    [
        opt_propars_convex_opt,
        opt_propars_trust_region,
        opt_propars_self_consistent,
        opt_propars_diis,
        # Newton failed
        # opt_propars_fixed_points_newton,
    ],
)
def test_optimization_methods(case, opt_propars):
    density_cutoff = 1e-15
    threshold = 1e-8

    populations, exponents = case
    rgrid = _setup_rgrid()
    r = rgrid.points

    gauss_funcs = [GaussianFunction(c, alpha) for c, alpha in zip(populations, exponents)]
    radii_max = max(func.get_cutoff_radius(density_cutoff) for func in gauss_funcs)
    r = r[r <= radii_max]
    rho = sum(func.compute(r) for func in gauss_funcs)
    check_rho(rho)
    assert np.sum(populations) == pytest.approx(1.0)

    # Setup grid
    local_rgrid = rgrid.get_localgrid(0.0, radii_max)
    # Initialize parameters
    propars = np.array([1.0 / len(exponents)] * len(exponents))

    # bs_funcs = np.asarray(gauss_funcs)
    bs_funcs = np.asarray([func.compute(r) for func in gauss_funcs])
    kwargs = {}
    if opt_propars == opt_propars_diis:
        kwargs["diis_size"] = 10

    local_r = local_rgrid.points
    local_weights = 4 * np.pi * local_r**2 * local_rgrid.weights

    # Optimization
    opt_propars = opt_propars(
        bs_funcs,
        rho,
        propars,
        local_r,
        local_weights,
        threshold,
        density_cutoff,
        logger,
        **kwargs,
    )

    opt_gauss_funcs = [GaussianFunction(c, alpha) for c, alpha in zip(opt_propars, exponents)]
    opt_rho = sum(func.compute(r) for func in opt_gauss_funcs)
    check_rho(opt_rho)
    assert np.sum(opt_propars) == pytest.approx(1.0, abs=1e-1)
