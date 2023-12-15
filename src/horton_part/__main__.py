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
"""Prepare inputs for denspart with HORTON3 modules.

This implementation makes some ad hoc choices on the molecular integration
grid, which should be revised in future. For now, this is something that is
just supposed to just work. The code is not tuned for precision nor for
efficiency.

This module is far from polished and is currently only used for prototyping:

- The integration grid is not final. It may have precision issues and it is not
  pruned. Furthermore, all atoms are given the same atomic grid, which is far
  from optimal and the Becke partitioning is used, which can be improved upon.

- Only the spin-summed density is computed, using the post-hf 1RDM if it is
  present. The spin-difference density is ignored.

- It is slow.

"""
import sys


def main():
    pass


if __name__ == "__main__":
    sys.exit(main())
