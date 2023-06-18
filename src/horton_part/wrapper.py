import horton_grid
import numpy as np

__all__ = ["solve_poisson_becke", "log", "biblio", "eval_spline_grid"]


solve_poisson_becke = horton_grid.solve_poisson_becke
log = horton_grid.log
biblio = horton_grid.biblio


def eval_spline_grid(spline, grid, center):
    r = np.linalg.norm(grid.points - center, axis=1)
    return spline(r)
