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
"""Becke partitioning"""


import numpy as np
from grid.becke import BeckeWeights

from .core.base import WPart
from .core.logging import deflist
from .utils import angstrom, radius_becke, radius_covalent

__all__ = ["BeckeWPart"]


class BeckeWPart(WPart):
    """Becke partitioning with Becke-Lebedev grids"""

    name = "b"
    options = ["lmax", "k"]
    linear = True

    def __init__(
        self,
        coordinates,
        numbers,
        pseudo_numbers,
        grid,
        moldens,
        spindens=None,
        local=True,
        lmax=3,
        k=3,
        logger=None,
    ):
        """
        **Optional arguments:** (that are not defined in ``WPart``)

        k
             The order of the polynomials used in the Becke partitioning.
        """
        self._k = k
        WPart.__init__(
            self,
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            local,
            lmax,
            logger,
        )

    def _init_log_scheme(self):
        self.logger.info(" Initialized: %s" % self.__class__.__name__)
        deflist(
            self.logger,
            [
                (" Scheme", "Becke"),
                (" Switching function", "k=%i" % self._k),
            ],
        )
        # biblio.cite("becke1988_multicenter", "the use of Becke partitioning")
        # biblio.cite(
        #     "slater1964", "the Brag-Slater radii used in the Becke partitioning"
        # )

    def update_at_weights(self):
        self.logger.info("Computing Becke weights.")

        # The list of radii is constructed to be as close as possible to
        # the original values used by Becke.
        radii = []
        for number in self.numbers:
            if number == 1:
                # exception defined in Becke's paper
                radius = 0.35 * angstrom
            else:
                radius = radius_becke[number]
                if radius is None:
                    # for cases not covered by Brag-Slater
                    radius = radius_covalent[number]
            radii.append(radius)
        radii = np.array(radii)

        # for new API
        radii_dict = {int(_atm): _radius for _atm, _radius in zip(self.numbers, radii)}
        bw_helper = BeckeWeights(radii_dict, self._k)

        # Actual work
        for index in range(self.natom):
            grid = self.get_grid(index)
            at_weights = self.cache.load("at_weights", index, alloc=grid.size)[0]
            at_weights[:] = bw_helper.compute_atom_weight(
                grid.points, self.coordinates, self.numbers, index
            )

    @property
    def k(self):
        """The order of the Becke switching function."""
        return self._k
