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
import numpy as np

__all__ = ["bfgs"]


def bfgs(df, s, olddf, oldH):
    """
    BFGS method to construct the inversion of the Hessian matrix

    Parameters
    ----------
    df
    s
    olddf
    oldH

    Returns
    -------
    np.ndarray
        The inverse of Hessian matrix.

    """
    y = df - olddf
    sy = s @ y
    H = (
        oldH
        + (sy + np.einsum("i,ij,j->", y, oldH, y)) * np.einsum("i,j->ij", s, s) / sy**2
        - (oldH @ np.einsum("i,j->ij", y, s) + np.einsum("i,j->ij", s, y) @ oldH) / sy
    )
    return H
