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
import os
import pkg_resources
from glob import glob

__all__ = ["context", "Context"]


class Context(object):
    """Finds out where the data directory is located etc.

    The data directory contains data files with standard basis sets and
    pseudo potentials.
    """

    def __init__(self):
        # Determine data directory (also for in-place build)
        self.data_dir = pkg_resources.resource_filename(__name__, "data")

        # Check if directories exist
        if not pkg_resources.resource_isdir(__name__, "data"):
            raise IOError(
                "Can not find the data files. The directory %s does not exist."
                % self.data_dir
            )

    def get_fn(self, filename):
        """Return the full path to the given filename in the data directory."""
        return os.path.join(self.data_dir, filename)

    def glob(self, pattern):
        """Return all files in the data directory that match the given pattern."""
        return glob(self.get_fn(pattern))


context = Context()
