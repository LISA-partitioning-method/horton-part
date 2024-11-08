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
"""Generalize Minimal Basis Iterative Stockholder (GMBIS) partitioning"""


import logging

import numpy as np
from scipy.special import gamma

from .core.iterstock import AbstractISAWPart
from .core.logging import deflist
from .mbis import _get_nshell
from .utils import check_pro_atom_parameters

__all__ = ["GMBISWPart", "_get_initial_gmbis_propars"]

logger = logging.getLogger(__name__)


# def _get_nshell(number):
#     noble = np.array([2, 10, 18, 36, 54, 86, 118])
#     return noble.searchsorted(number) + 1


def _get_initial_gmbis_propars(number, exp_n_dict):
    nshell = _get_nshell(number)
    propars = np.zeros(3 * nshell, float)
    S0 = 2.0 * number
    if nshell > 1:
        S1 = 2.0
        alpha = (S1 / S0) ** (1.0 / (nshell - 1))
    else:
        alpha = 1.0
    nel_in_shell = np.array([2.0, 8.0, 8.0, 18.0, 18.0, 32.0, 32.0])
    for ishell in range(nshell):
        propars[3 * ishell] = nel_in_shell[ishell]
        propars[3 * ishell + 1] = S0 * alpha**ishell
        if (number, ishell) in exp_n_dict:
            propars[3 * ishell + 2] = exp_n_dict[(number, ishell)]
        else:
            propars[3 * ishell + 2] = 1.0
    propars[-3] = number - propars[:-3:3].sum()
    return propars


def _opt_gmbis_propars(rho, propars, rgrid, threshold, density_cutoff=1e-15):
    assert len(propars) % 3 == 0
    nshell = len(propars) // 3
    r = rgrid.points
    terms = np.zeros((nshell, len(r)), float)
    oldpro = None
    logger.debug("            Iter.    Change    ")
    logger.debug("            -----    ------    ")
    pop = rgrid.integrate(4 * np.pi * r**2, rho)
    for irep in range(1000):
        # compute the contributions to the pro-atom
        for ishell in range(nshell):
            N = propars[3 * ishell]
            S = propars[3 * ishell + 1]
            n = propars[3 * ishell + 2]
            # terms[ishell] = N * S**3 * np.exp(-S * r) / (8 * np.pi)
            func_g = n * S ** (3 / n) * np.exp(-S * r**n) / (4 * np.pi * gamma(3.0 / n))
            terms[ishell] = func_g

        # pro = terms.sum(axis=0)
        pro = np.sum(terms * propars[::3, None], axis=0)
        sick = (rho < density_cutoff) | (pro < density_cutoff)
        with np.errstate(all="ignore"):
            lnpro = np.log(pro)
            ratio = rho / pro
            lnratio = np.log(rho) - np.log(pro)
        lnpro[sick] = 0.0
        ratio[sick] = 0.0
        lnratio[sick] = 0.0

        # pro = np.clip(pro, 1e-100, np.inf)
        # transform to partitions
        # terms *= rho / pro
        # terms *= ratio
        terms_ratio = terms * ratio
        # the partitions and the updated parameters
        for ishell in range(nshell):
            N = propars[3 * ishell]
            S = propars[3 * ishell + 1]
            n = propars[3 * ishell + 2]

            m0 = rgrid.integrate(4 * np.pi * r**2, terms_ratio[ishell] * N)
            m1 = rgrid.integrate(4 * np.pi * r**2, terms_ratio[ishell], r**n)

            propars[3 * ishell] = m0
            if np.isclose(m1, 0.0):
                propars[3 * ishell + 1] = 1e-5
            else:
                propars[3 * ishell + 1] = 3 / (m1 * n)

        # check for convergence
        if oldpro is None:
            change = 1e100
        else:
            error = oldpro - pro
            change = np.sqrt(rgrid.integrate(4 * np.pi * r**2, error, error))

        logger.debug(f"            {irep+1:<4}    {change:.3e}")

        if change < threshold:
            if not np.isclose(pop, np.sum(propars[0::3]), atol=1e-4):
                RuntimeWarning("The sum of propars are not equal to the atomic pop.")
            check_pro_atom_parameters(
                propars,
                pro_atom_density=pro,
                check_monotonicity=True,
                check_negativity=True,
                check_propars_negativity=True,
            )
            return propars
        oldpro = pro
    # logger.warn("NLIS not converged!")
    return propars
    # assert False


class GMBISWPart(AbstractISAWPart):
    """Generalize Minimal Basis Iterative Stockholder (MBIS)"""

    name = "gmbis"

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
        logger=None,
        threshold=1e-6,
        maxiter=500,
        inner_threshold=1e-8,
        radius_cutoff=np.inf,
        exp_n_dict=1.0,
        **kwargs,
    ):
        r"""
        Initial function.

        **Optional arguments:** (that are not defined in :class:`horton_part.core.itertock.AbstractISAWPart`)

        Parameters
        ----------
        exp_n_dict : dict
            The power of radial distance :math:`r` in exponential function :math:`exp^{-\alpha r^n}`.

        """
        self._exp_n_dict = exp_n_dict

        super().__init__(
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            local=local,
            lmax=lmax,
            logger=logger,
            threshold=threshold,
            maxiter=maxiter,
            inner_threshold=inner_threshold,
            radius_cutoff=radius_cutoff,
        )

    def _init_log_scheme(self):
        logger.info("Initialized: %s" % self.__class__.__name__)
        deflist(
            logger,
            [
                ("Scheme", "Generalize Minimal Basis Iterative Stockholder (GMBIS)"),
                ("Outer loop convergence threshold", "%.1e" % self._threshold),
                (
                    "Inner loop convergence threshold",
                    "%.1e" % self._inner_threshold,
                ),
                ("Maximum iterations", self._maxiter),
            ],
        )
        # biblio.cite("verstraelen2016", "the use of MBIS partitioning")

    def get_rgrid(self, iatom):
        """Get radial grid for `iatom` atom."""
        return self.get_grid(iatom).rgrid

    def get_proatom_rho(self, iatom, propars=None, **kwargs):
        """Get pro-atom density for atom `iatom`.

        If `propars` is `None`, the cache values are used; otherwise, the `propars` are used.

        Parameters
        ----------
        iatom: int
            The index of atom `iatom`.
        propars: np.array
            The pro-atom parameters.

        """
        if propars is None:
            propars = self.cache.load("propars")
        rgrid = self.get_rgrid(iatom)
        r = rgrid.points
        y = np.zeros(len(r), float)
        d = np.zeros(len(r), float)
        my_propars = propars[self._ranges[iatom] : self._ranges[iatom + 1]]
        for ishell in range(self._nshells[iatom]):
            N, S, n = my_propars[3 * ishell : 3 * ishell + 3]
            f = N * n * S ** (3 / n) * np.exp(-S * r**n) / (4 * np.pi * gamma(3.0 / n))
            y += f
            d -= N * S * n * r ** (n - 1) * f
        return y, d

    def eval_proatom(self, index, output, grid):
        """Evaluate function on a local grid.

        The size of the local grid is specified by the radius of the sphere where the local grid is considered.
        For example, when the radius is `np.inf`, the grid corresponds to the whole molecular grid.

        Parameters
        ----------
        index: int
            The index of an atom in the molecule.
        output: np.array
            The size of `output` should be the same as the size of the local grid.
        grid: np.array
            The local grid.

        """
        propars = self.cache.load("propars")
        r = self.radial_distances[index]
        y = np.zeros(len(r), float)
        # d = np.zeros(len(r), float)
        my_propars = propars[self._ranges[index] : self._ranges[index + 1]]
        for ishell in range(self._nshells[index]):
            N, S, n = my_propars[3 * ishell : 3 * ishell + 3]
            f = N * n * S ** (3 / n) * np.exp(-S * r**n) / (4 * np.pi * gamma(3.0 / n))
            y += f
        output[:] = y

    def _init_propars(self):
        AbstractISAWPart._init_propars(self)
        self._ranges = [0]
        self._nshells = []
        for iatom in range(self.natom):
            nshell = _get_nshell(self.numbers[iatom])
            self._ranges.append(self._ranges[-1] + 3 * nshell)
            self._nshells.append(nshell)
        ntotal = self._ranges[-1]
        propars = self.cache.load("propars", alloc=ntotal, tags="o")[0]
        for iatom in range(self.natom):
            propars[self._ranges[iatom] : self._ranges[iatom + 1]] = _get_initial_gmbis_propars(
                self.numbers[iatom], self._exp_n_dict
            )
        return propars

    def _update_propars_atom(self, iatom):
        # compute spherical average
        atgrid = self.get_grid(iatom)
        rgrid = atgrid.rgrid
        dens = self.get_moldens(iatom)
        at_weights = self.cache.load(f"at_weights_{iatom}")
        spline = atgrid.spherical_average(at_weights * dens)
        points = rgrid.points
        spherical_average = spline(points)
        # spherical_average = np.clip(spline(rgrid.points), 1e-100, np.inf)
        self.cache.dump(f"radial_points_{iatom}", points, tags="o")
        self.cache.dump(f"spherical_average_{iatom}", spherical_average, tags="o")
        self.cache.dump(f"radial_weights_{iatom}", rgrid.weights, tags="o")

        # assign as new propars
        my_propars = self.cache.load("propars")[self._ranges[iatom] : self._ranges[iatom + 1]]
        my_propars[:] = _opt_gmbis_propars(
            spherical_average, my_propars.copy(), rgrid, self._inner_threshold
        )

        # avoid too large r
        # r = np.clip(rgrid.points, 1e-100, 1e10)

        # compute the new charge
        # 4 * np.pi * rgrid.points ** 2 * spherical_average
        pseudo_population = rgrid.integrate(4 * np.pi * points**2, spherical_average)
        charges = self.cache.load("charges", alloc=self.natom, tags="o")[0]
        charges[iatom] = self.pseudo_numbers[iatom] - pseudo_population

    def _finalize_propars(self):
        AbstractISAWPart._finalize_propars(self)
        propars = self.cache.load("propars")
        core_charges = []
        valence_charges = []
        valence_widths = []
        for iatom in range(self.natom):
            my_propars = propars[self._ranges[iatom] : self._ranges[iatom + 1]]
            valence_charges.append(-my_propars[-3])
            valence_widths.append(1.0 / my_propars[-2])
        valence_charges = np.array(valence_charges)
        valence_widths = np.array(valence_widths)
        core_charges = self._cache.load("charges") - valence_charges
        self.cache.dump("core_charges", core_charges, tags="o")
        self.cache.dump("valence_charges", valence_charges, tags="o")
        self.cache.dump("valence_widths", valence_widths, tags="o")
