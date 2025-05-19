# HORTON-PART: molecular density partition schemes based on HORTON package.
# Copyright (C) 2023-2025 The HORTON-PART Development Team
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
"""Minimal Basis Iterative Stockholder (MBIS) partitioning"""


import logging

import numpy as np

from .core.iterstock import AbstractISAWPart
from .core.logging import deflist
from .utils import check_pro_atom_parameters_non_neg_pars

__all__ = ["MBISWPart", "get_nshell", "get_initial_mbis_propars", "update_propars_atom"]

logger = logging.getLogger(__name__)


def get_nshell(number: int):
    """Get the number of shell of an element.

    Parameters
    ----------
    number
        The atomic number.

    """
    noble = np.array([2, 10, 18, 36, 54, 86, 118])
    return noble.searchsorted(number) + 1


def get_initial_mbis_propars(number: int):
    """Set up initial pro-atom parameters for the MBIS model.

    Parameters
    ----------
    number : int
        Atomic number, denoted by :math:`Z`, of the element for which to generate
        initial pro-atom parameters.

    Returns
    -------
    np.ndarray
        A 1D numpy array containing the initial parameters for the MBIS model
        corresponding to the specified atomic number.

    """
    nshell = get_nshell(number)
    propars = np.zeros(2 * nshell, float)
    S0 = 2.0 * number
    if nshell > 1:
        S1 = 2.0
        alpha = (S1 / S0) ** (1.0 / (nshell - 1))
    else:
        alpha = 1.0
    nel_in_shell = np.array([2.0, 8.0, 8.0, 18.0, 18.0, 32.0, 32.0])
    for ishell in range(nshell):
        propars[2 * ishell] = nel_in_shell[ishell]
        propars[2 * ishell + 1] = S0 * alpha**ishell
    propars[-2] = number - propars[:-2:2].sum()
    return propars


def opt_mbis_propars(
    rho, propars, weights, radial_dist, threshold, density_cutoff=1e-15, logger=logger
):
    r"""
    Optimize pro-atom parameters for the MBIS model.

    Parameters
    ----------
    rho : np.ndarray
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
    assert len(propars) % 2 == 0
    nshell = len(propars) // 2
    r = radial_dist
    terms = np.zeros((nshell, len(r)), float)
    oldpro = None
    logger.debug("            Iter.    Change    ")
    logger.debug("            -----    ------    ")
    pop = np.einsum("p,p->", weights, rho)
    for irep in range(2000):
        # compute the contributions to the pro-atom
        for ishell in range(nshell):
            N = propars[2 * ishell]
            S = propars[2 * ishell + 1]
            terms[ishell] = N * S**3 * np.exp(-S * r) / (8 * np.pi)

        pro = terms.sum(axis=0)
        sick = (rho < density_cutoff) | (pro < density_cutoff)
        with np.errstate(all="ignore"):
            lnpro = np.log(pro)
            ratio = rho / pro
            lnratio = np.log(rho) - np.log(pro)
        lnpro[sick] = 0.0
        ratio[sick] = 0.0
        lnratio[sick] = 0.0

        terms *= ratio
        # the partitions and the updated parameters
        for ishell in range(nshell):
            m0 = np.einsum("p,p->", weights, terms[ishell])
            m1 = np.einsum("p,p,p->", weights, terms[ishell], r)
            propars[2 * ishell] = m0
            propars[2 * ishell + 1] = 3 * m0 / m1
        # check for convergence
        if oldpro is None:
            change = 1e100
        else:
            error = oldpro - pro
            change = np.sqrt(np.einsum("p,p,p->", weights, error, error))

        logger.debug(f"            {irep+1:<4}    {change:.3e}")

        if change < threshold:
            if not np.isclose(pop, np.sum(propars[0::2]), atol=1e-4):
                logger.warning("The sum of propars are not equal to the atomic pop.")
            check_pro_atom_parameters_non_neg_pars(propars, pro_atom_density=pro, logger=logger)
            return propars
        oldpro = pro
    logger.warning("MBIS not converged, but still go ahead!")
    return propars
    # assert False


def update_propars_atom(part, iatom, solver):
    """Update pro-atom parameters."""
    at_weights = part.cache.load(f"at_weights_{iatom}")
    if part.on_molgrid:
        rhoa = at_weights * part._moldens
        weights = part.grid.weights
        radial_dist = part.radial_distances[iatom]
    else:
        # compute spherical average
        atgrid = part.get_grid(iatom)
        rgrid = atgrid.rgrid
        dens = part.get_moldens(iatom)
        spline = atgrid.spherical_average(at_weights * dens)
        radial_weights = rgrid.weights
        radial_dist = rgrid.points
        weights = 4 * np.pi * radial_dist**2 * radial_weights
        # spherical average
        rhoa = spline(radial_dist)
        part.cache.dump(f"radial_points_{iatom}", radial_dist, tags="o")
        part.cache.dump(f"spherical_average_{iatom}", rhoa, tags="o")
        part.cache.dump(f"radial_weights_{iatom}", radial_weights, tags="o")

    # assign as new propars
    my_propars = part.cache.load("propars")[part._ranges[iatom] : part._ranges[iatom + 1]]
    my_propars[:] = solver(
        rhoa,
        my_propars.copy(),
        weights=weights,
        radial_dist=radial_dist,
        threshold=part._inner_threshold,
        logger=part.logger,
    )

    # compute the new charge
    pseudo_population = np.einsum("p,p->", weights, rhoa)
    charges = part.cache.load("charges", alloc=part.natom, tags="o")[0]
    charges[iatom] = part.pseudo_numbers[iatom] - pseudo_population


class MBISWPart(AbstractISAWPart):
    """Minimal Basis Iterative Stockholder (MBIS)"""

    name = "mbis"

    def _init_log_scheme(self):
        logger.info("Initialized: %s" % self.__class__.__name__)
        deflist(
            logger,
            [
                ("Scheme", "Minimal Basis Iterative Stockholder (MBIS)"),
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
        if self.only_use_molgrid:
            raise NotImplementedError
        else:
            return self.get_grid(iatom).rgrid

    def get_proatom_rho(self, iatom: int, propars=None, **kwargs):
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
            r = self.get_rgrid(iatom).points
        y = np.zeros(len(r), float)
        d = np.zeros(len(r), float)
        my_propars = propars[self._ranges[iatom] : self._ranges[iatom + 1]]
        for ishell in range(self._nshells[iatom]):
            N, S = my_propars[2 * ishell : 2 * ishell + 2]
            f = N * S**3 * np.exp(-S * r) / (8 * np.pi)
            y += f
            d -= S * f
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
            N, S = my_propars[2 * ishell : 2 * ishell + 2]
            f = N * S**3 * np.exp(-S * r) / (8 * np.pi)
            y += f
            # d -= S * f
        output[:] = y

    def _init_propars(self):
        # AbstractISAWPart._init_propars(self)
        self._ranges = [0]
        self._nshells = []
        for iatom in range(self.natom):
            nshell = get_nshell(self.numbers[iatom])
            self._ranges.append(self._ranges[-1] + 2 * nshell)
            self._nshells.append(nshell)
        ntotal = self._ranges[-1]
        propars = self.cache.load("propars", alloc=ntotal, tags="o")[0]
        for iatom in range(self.natom):
            propars[self._ranges[iatom] : self._ranges[iatom + 1]] = get_initial_mbis_propars(
                self.numbers[iatom]
            )
        return propars

    def _update_propars_atom(self, iatom):
        update_propars_atom(self, iatom, solver=opt_mbis_propars)
        # at_weights = self.cache.load(f"at_weights_{iatom}")
        # if self.on_molgrid:
        #     rhoa = at_weights * self._moldens
        #     weights = self.grid.weights
        #     radial_dist = self.radial_distances[iatom]
        # else:
        #     # compute spherical average
        #     atgrid = self.get_grid(iatom)
        #     rgrid = atgrid.rgrid
        #     dens = self.get_moldens(iatom)
        #     spline = atgrid.spherical_average(at_weights * dens)
        #     radial_weights = rgrid.weights
        #     radial_dist = rgrid.points
        #     weights = 4 * np.pi * radial_dist**2 * radial_weights
        #     # spherical average
        #     rhoa = spline(radial_dist)
        #     self.cache.dump(f"radial_points_{iatom}", radial_dist, tags="o")
        #     self.cache.dump(f"spherical_average_{iatom}", rhoa, tags="o")
        #     self.cache.dump(f"radial_weights_{iatom}", radial_weights, tags="o")

        # # assign as new propars
        # my_propars = self.cache.load("propars")[
        #     self._ranges[iatom] : self._ranges[iatom + 1]
        # ]
        # my_propars[:] = opt_mbis_propars(
        #     rhoa,
        #     my_propars.copy(),
        #     weights=weights,
        #     radial_dist=radial_dist,
        #     threshold=self._inner_threshold,
        #     logger=self.logger,
        # )

        # # compute the new charge
        # pseudo_population = np.einsum("p,p->", weights, rhoa)
        # charges = self.cache.load("charges", alloc=self.natom, tags="o")[0]
        # charges[iatom] = self.pseudo_numbers[iatom] - pseudo_population

    def _finalize_propars(self):
        AbstractISAWPart._finalize_propars(self)
        propars = self.cache.load("propars")
        core_charges = []
        valence_charges = []
        valence_widths = []
        for iatom in range(self.natom):
            my_propars = propars[self._ranges[iatom] : self._ranges[iatom + 1]]
            valence_charges.append(-my_propars[-2])
            valence_widths.append(1.0 / my_propars[-1])
        valence_charges = np.array(valence_charges)
        valence_widths = np.array(valence_widths)
        core_charges = self._cache.load("charges") - valence_charges
        self.cache.dump("core_charges", core_charges, tags="o")
        self.cache.dump("valence_charges", valence_charges, tags="o")
        self.cache.dump("valence_widths", valence_widths, tags="o")
