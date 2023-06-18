import numpy as np

__all__ = ["eval_spline_grid"]


def eval_spline_grid(spline, grid, center):
    r = np.linalg.norm(grid.points - center, axis=1)
    return spline(r)
