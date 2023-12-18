# -*- coding: utf-8 -*-
# HORTON-PART: GRID for Helpful Open-source Research TOol for N-fermion systems.
# Copyright (C) 2011-2023 The HORTON-PART Development Team
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
"""Hirshfeld partitioning"""

from .core.cache import just_once
from .core.stockholder import AbstractStockholderWPart
from .core.log import log, biblio


__all__ = ["HirshfeldWPart", "check_proatomdb", "do_dispersion"]


def check_proatomdb(numbers, pseudo_numbers, proatomdb):
    # Check if the same pseudo numbers (effective core charges) are used for the
    # molecule and the proatoms.
    for i in range(len(numbers)):
        number = numbers[i]
        pseudo_number = pseudo_numbers[i]
        pseudo_number_expected = proatomdb.get_record(number, 0).pseudo_number
        if pseudo_number_expected != pseudo_number:
            raise ValueError(
                "The pseudo number of atom %i does not match with the proatom database (%i!=%i)"
                % (i, pseudo_number, pseudo_number_expected)
            )


def do_dispersion(part):
    if part.lmax < 3:
        print(
            "!WARNING! Skip computing dispersion coefficients because lmax=%i<3"
            % part.lmax
        )
        biblio.cite(
            "tkatchenko2009",
            "the method to evaluate atoms-in-molecules C6 parameters",
        )
        biblio.cite("chu2004", "the reference C6 parameters of isolated atoms")
        biblio.cite("yan1996", "the isolated hydrogen C6 parameter")

    # reference C6 values in atomic units
    ref_c6s = {
        1: 6.499,
        2: 1.42,
        3: 1392.0,
        4: 227.0,
        5: 99.5,
        6: 46.6,
        7: 24.2,
        8: 15.6,
        9: 9.52,
        10: 6.20,
        11: 1518.0,
        12: 626.0,
        13: 528.0,
        14: 305.0,
        15: 185.0,
        16: 134.0,
        17: 94.6,
        18: 64.2,
        19: 3923.0,
        20: 2163.0,
        21: 1383.0,
        22: 1044.0,
        23: 832.0,
        24: 602.0,
        25: 552.0,
        26: 482.0,
        27: 408.0,
        28: 373.0,
        29: 253.0,
        30: 284.0,
        31: 498.0,
        32: 354.0,
        33: 246.0,
        34: 210.0,
        35: 162.0,
        36: 130.0,
        37: 4769.0,
        38: 3175.0,
        49: 779.0,
        50: 659.0,
        51: 492.0,
        52: 445.0,
        53: 385.0,
    }

    volumes, new_volumes = part._cache.load("volumes", alloc=part.natom, tags="o")
    volume_ratios, new_volume_ratios = part._cache.load(
        "volume_ratios", alloc=part.natom, tags="o"
    )
    c6s, new_c6s = part._cache.load("c6s", alloc=part.natom, tags="o")

    if new_volumes or new_volume_ratios or new_c6s:
        part.do_moments()
        radial_moments = part._cache.load("radial_moments")

        print("Computing atomic dispersion coefficients.")

        for i in range(part.natom):
            n = part.numbers[i]
            volumes[i] = radial_moments[i, 3]
            ref_volume = part.proatomdb.get_record(n, 0).get_moment(3)
            volume_ratios[i] = volumes[i] / ref_volume
            if n in ref_c6s:
                c6s[i] = (volume_ratios[i]) ** 2 * ref_c6s[n]
            else:
                # This is just used to indicate that no value is available.
                c6s[i] = -1


class HirshfeldWPart(AbstractStockholderWPart):
    """Hirshfeld partitioning with Becke-Lebedev grids"""

    name = "h"

    def __init__(
        self,
        coordinates,
        numbers,
        pseudo_numbers,
        grid,
        moldens,
        proatomdb,
        spindens=None,
        local=True,
        lmax=3,
    ):
        """
        **Arguments:** (that are not defined in ``WPart``)

        proatomdb
             In instance of ProAtomDB that contains all the reference atomic
             densities.
        """
        check_proatomdb(numbers, pseudo_numbers, proatomdb)
        self._proatomdb = proatomdb
        AbstractStockholderWPart.__init__(
            self,
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            local,
            lmax,
        )

    def _init_log_scheme(self):
        if log.do_medium:
            log("Initialized: %s" % self.__class__.__name__)
            log.deflist(
                [
                    ("Scheme", "Hirshfeld"),
                    ("Proatomic DB", self.proatomdb),
                ]
            )
            biblio.cite("hirshfeld1977", "the use of Hirshfeld partitioning")

    def _get_proatomdb(self):
        return self._proatomdb

    proatomdb = property(_get_proatomdb)

    def get_rgrid(self, index):
        number = self.numbers[index]
        return self.proatomdb.get_rgrid(number)

    def get_proatom_rho(self, index):
        return self.proatomdb.get_rho(self.numbers[index], do_deriv=True)

    @just_once
    def do_dispersion(self):
        do_dispersion(self)
