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
"""Non-Linear approximation of Iterative Stockholder (NLIS) partitioning"""


import logging

import numpy as np
from scipy.special import gamma

from .core.iterstock import AbstractISAWPart
from .core.logging import deflist
from .mbis import get_nshell, update_propars_atom
from .utils import check_pro_atom_parameters_non_neg_pars

__all__ = [
    "NLISWPart",
    "get_initial_nlis_propars",
    "get_nlis_nshell",
    "opt_nlis_propars",
]

logger = logging.getLogger(__name__)


def get_nlis_nshell(number, nshell_dict):
    """Get the number of shell of an element.

    Parameters
    ----------
    number
        The atomic number.
    nshell_dict: dict
        A dict for the number of shells for each element.

    """
    return nshell_dict.get(number, get_nshell(number))


def get_initial_nlis_propars(number, exp_n_dict, nshell_dict, logger=None):
    """Set up initial pro-atom parameters for the NLIS model.

    Parameters
    ----------
    number : int
        Atomic number, denoted by :math:`Z`, of the element for which to generate
        initial pro-atom parameters.
    exp_n_dict: dict
        A dict of exponents for each (number, ishell).
    nshell_dict: dict
        A dict for the number of shells for each element.
    logger : logging.Logger
        Logger object for capturing logging information.

    Returns
    -------
    np.ndarray
        1D array of initial pro-atom parameters.

    """
    nbs = get_nlis_nshell(number, nshell_dict)
    propars = np.ones(3 * nbs, float)
    S0 = 2.0 * number
    if nbs > 1:
        S1 = 0.5
        alpha = (S1 / S0) ** (1.0 / (nbs - 1))
    else:
        alpha = 1.0
    for ibs in range(nbs):
        # initial c_ak
        propars[3 * ibs] = number / nbs
        # alpha
        propars[3 * ibs + 1] = S0 * alpha**ibs
        # n
        if (number, ibs) in exp_n_dict:
            propars[3 * ibs + 2] = exp_n_dict[(number, ibs)]
        else:
            propars[3 * ibs + 2] = 1.0
    return propars


def opt_nlis_propars(
    rhoa, propars, weights, radial_dist, threshold, density_cutoff=1e-15, logger=None
):
    r"""
    Optimize pro-atom parameters for the MBIS model.

    Parameters
    ----------
    rhoa : np.ndarray
        Spherically averaged AIM density (1D) or AIM density (3D) of atom `a`.
    propars : np.ndarray
        Initial pro-atom parameters for optimization.
    weights : np.ndarray
        Integrand weights; if a spherically averaged density is used, includes :math:`4 \pi r^2`.
    radial_dist : np.ndarray
        Array of radial distances with respect to the coordinate of atom `a`.
    threshold : float
        Values of density below this threshold are treated as zero.
    density_cutoff : float, optional
        Density values below this cutoff are treated as zero (default is 1e-15).
    logger : logging.Logger
        Logger object for capturing logging information.

    Returns
    -------
    np.ndarray
        1D array of updated pro-atom parameters.

    Raises
    ------
    RuntimeError
        Raised if the sum of pro-atom parameters does not equal the population of atom `a`.

    """
    assert len(propars) % 3 == 0
    nshell = len(propars) // 3
    r = radial_dist
    terms = np.zeros((nshell, len(r)), float)
    oldpro = None
    logger.debug("            Iter.    Change    ")
    logger.debug("            -----    ------    ")
    pop = np.einsum("p,p->", weights, rhoa)
    for irep in range(2000):
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
        sick = (rhoa < density_cutoff) | (pro < density_cutoff)
        with np.errstate(all="ignore"):
            lnpro = np.log(pro)
            ratio = rhoa / pro
            lnratio = np.log(rhoa) - np.log(pro)
        lnpro[sick] = 0.0
        ratio[sick] = 0.0
        lnratio[sick] = 0.0

        terms_ratio = terms * ratio
        # the partitions and the updated parameters
        for ishell in range(nshell):
            N = propars[3 * ishell]
            S = propars[3 * ishell + 1]
            n = propars[3 * ishell + 2]

            m0 = np.einsum("p,p->", weights, terms_ratio[ishell] * N)
            m1 = np.einsum("p,p,p->", weights, terms_ratio[ishell], r**n)

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
            change = np.sqrt(np.einsum("p,p,p->", weights, error, error))

        logger.debug(f"            {irep+1:<4}    {change:.3e}")

        if change < threshold:
            if not np.isclose(pop, np.sum(propars[0::3]), atol=1e-4):
                logger.warning("The sum of propars are not equal to the atomic pop.")
            check_pro_atom_parameters_non_neg_pars(propars, pro_atom_density=pro, logger=logger)
            return propars
        oldpro = pro
    logger.warning("NLIS not converged, but still go ahead!")
    return propars
    # assert False


class NLISWPart(AbstractISAWPart):
    """Non-Linear approximation of Iterative Stockholder (NLIS)"""

    name = "nlis"

    def __init__(
        self,
        coordinates,
        numbers,
        pseudo_numbers,
        grid,
        moldens,
        spindens=None,
        lmax=3,
        logger=None,
        threshold=1e-6,
        maxiter=500,
        inner_threshold=1e-8,
        exp_n_dict=1.0,
        nshell_dict=None,
        grid_type=1,
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
        self._nshell_dict = nshell_dict or {}

        super().__init__(
            coordinates,
            numbers,
            pseudo_numbers,
            grid,
            moldens,
            spindens,
            lmax=lmax,
            logger=logger,
            threshold=threshold,
            maxiter=maxiter,
            inner_threshold=inner_threshold,
            grid_type=grid_type,
        )

    def _init_log_scheme(self):
        logger.info("Initialized: %s" % self.__class__.__name__)
        deflist(
            logger,
            [
                ("Scheme", "Non-Linear approximation of Iterative Stockholder (NLIS)"),
                ("Outer loop convergence threshold", "%.1e" % self._threshold),
                (
                    "Inner loop convergence threshold",
                    "%.1e" % self._inner_threshold,
                ),
                ("Maximum iterations", self._maxiter),
            ],
        )
        # biblio.cite("verstraelen2016", "the use of NLIS partitioning")

    def get_rgrid(self, iatom):
        """Get radial grid for `iatom` atom."""
        if self.only_use_molgrid:
            raise NotImplementedError
        else:
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
        if self.on_molgrid:
            r = self.radial_distances[iatom]
        else:
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
        """Evaluate function on the molecular grid.

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
        # AbstractISAWPart._init_propars(self)
        self._ranges = [0]
        self._nshells = []
        for iatom in range(self.natom):
            nshell = get_nlis_nshell(self.numbers[iatom], self._nshell_dict)
            self._ranges.append(self._ranges[-1] + 3 * nshell)
            self._nshells.append(nshell)
        ntotal = self._ranges[-1]
        propars = self.cache.load("propars", alloc=ntotal, tags="o")[0]
        for iatom in range(self.natom):
            propars[self._ranges[iatom] : self._ranges[iatom + 1]] = get_initial_nlis_propars(
                self.numbers[iatom],
                self._exp_n_dict,
                self._nshell_dict,
                logger=self.logger,
            )
        return propars

    def _update_propars_atom(self, iatom):
        update_propars_atom(self, iatom, solver=opt_nlis_propars)

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
