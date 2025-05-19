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
"""Base classes for (atoms-in-molecules) partitioning algorithms"""


import logging

import numpy as np
from grid import AtomGrid

from ..utils import DENSITY_CUTOFF, NEGATIVE_CUTOFF, POPULATION_CUTOFF, typecheck_geo
from .cache import Cache, JustOnceClass, just_once
from .logging import deflist, setup_logger

__all__ = ["Part", "WPart"]


class Part(JustOnceClass):
    name = None

    def __init__(
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
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        coordinates
             An array (N, 3) with centers for the atom-centered grids.
        numbers : 1D np.ndarray
             An array (N,) with atomic numbers.
        pseudo_numbers : 1D np.ndarray
             An array (N,) with effective charges. When set to None, this
             defaults to``numbers.astype(float)``.
        grid : Grid
             A Molecular integration grid. This must be a BeckeMolGrid
             instance with mode=='keep' or mode=='only'.
        moldens : 1D np.ndarray
             The spin-summed electron density on the grid.
        spindens : 1D np.ndarray or None
             The spin difference density on the grid.
        local : bool
             If ``True``: use the proper atomic grid for each AIM integral.
             If ``False``: use the entire molecular grid for each AIM integral.
        lmax : int
             The maximum angular momentum in multipole expansions.
        logger : logging.Logger
            A `logging.Logger` object.
        """
        # Init base class
        super().__init__()

        # Some type checking for first three arguments
        natom, coordinates, numbers, pseudo_numbers = typecheck_geo(
            coordinates, numbers, pseudo_numbers
        )
        self._natom = natom
        self._coordinates = coordinates
        self._numbers = numbers
        self._pseudo_numbers = pseudo_numbers

        # Assign remaining arguments as attributes
        self._grid = grid
        self._moldens = moldens
        self._spindens = spindens
        self._local = local
        self._lmax = lmax

        # Caching stuff, to avoid recomputation of earlier results
        self._cache = Cache()

        # Some screen logging
        self.logger = logger
        # Initialize the subgrids
        if local:
            self._init_subgrids()
        # self.biblio = []
        self._init_log_base()
        self._init_log_scheme()
        # self._init_log_memory()

    def __getitem__(self, key):
        return self.cache.load(key)

    def variables_stored_in_cache(self):
        """The properties stored in cache obj."""
        return list(self.cache.iterkeys())

    @property
    def natom(self):
        """The number of atoms in the molecule."""
        return self._natom

    @property
    def coordinates(self):
        r"""ndarray(M, 3) : Center/Atomic coordinates."""
        return self._coordinates

    @property
    def numbers(self):
        """Atomic numbers"""
        return self._numbers

    @property
    def pseudo_numbers(self):
        """Atomic charges."""
        return self._pseudo_numbers

    @property
    def nelec(self):
        """The number of electrons in the molecule."""
        return self.grid.integrate(self._moldens)

    @property
    def grid(self):
        """Molecular grid."""
        return self.get_grid()

    @property
    def local(self):
        """Whether local grids are included."""
        return self._local

    @property
    def lmax(self):
        """The maximum angular momentum index for moment calculations."""
        return self._lmax

    @property
    def cache(self):
        """Cache."""
        return self._cache

    def __clear__(self):
        self.clear()

    def clear(self):
        """Discard all cached results, e.g. because wfn changed"""
        JustOnceClass.clear(self)
        self.cache.clear()

    def get_grid(self, index=None):
        """Return an integration grid

        **Optional arguments:**

        index
             The index of the atom. If not given, a grid for the entire
             system is returned. If self.local is False, a full system grid
             is always returned.
        """
        if index is None or not self.local:
            return self._grid
        else:
            return self._subgrids[index]

    def get_moldens(self, index=None, output=None):
        """
        Retrieves the molecular electron density (moldens) on the atomic grid.

        This method converts the molecular electron density to the atomic grid specified
        by the index. If an output array is provided, the result is stored in that array.

        Parameters
        ----------
        index : int or None, optional
            The index of the atom for which the electron density is required.
            If None, electron density for all atoms is considered. Default is None.
        output : np.ndarray or None, optional
            An optional array to store the resulting electron density.
            If provided, the result is saved in this array. Default is None.

        Returns
        -------
        np.ndarray
            The molecular electron density on the atomic grid.
        """
        if index is None or not self.local:
            result = self._moldens
        else:
            result = self.to_atomic_grid(index, self._moldens)
        if output is not None:
            output[:] = result
        return result

    def get_spindens(self, index=None, output=None):
        """
        Retrieves the spin density (spindens) on the atomic grid.

        This method converts the spin density to the atomic grid specified by the index.
        If an output array is provided, the result is stored in that array.

        Parameters
        ----------
        index : int or None, optional
            The index of the atom for which the spin density is required.
            If None, spin density for all atoms is considered. Default is None.
        output : np.ndarray or None, optional
            An optional array to store the resulting spin density.
            If provided, the result is saved in this array. Default is None.

        Returns
        -------
        np.ndarray
            The spin density on the atomic grid.
        """
        if index is None or not self.local:
            result = self._spindens
        else:
            result = self.to_atomic_grid(index, self._spindens)
        if output is not None:
            output[:] = result
        return result

    def _init_subgrids(self):
        raise NotImplementedError

    def _init_log_base(self):
        raise NotImplementedError

    def _init_log_scheme(self):
        raise NotImplementedError

    def to_atomic_grid(self, index, data):
        """Load atomic contribution of molecular properties."""
        raise NotImplementedError

    def get_wcor(self, index):
        """Load correction of weights."""
        return 1.0

    def compute_pseudo_population(self, index):
        """Compute pseudo population"""
        grid = self.get_grid(index)
        dens = self.get_moldens(index)
        at_weights = self.cache.load(f"at_weights_{index}")
        return grid.integrate(at_weights, dens)

    @just_once
    def do_partitioning(self):
        """Run partitioning."""
        self.update_at_weights()

    do_partitioning.names = []

    def update_at_weights(self, *args, **kwargs):
        """Updates the at_weights arrays in the case (and all related arrays)"""
        raise NotImplementedError

    @just_once
    def do_populations(self):
        """Compute atomic populations."""
        populations, new = self.cache.load("populations", alloc=self.natom, tags="o")
        if new:
            self.do_partitioning()
            pseudo_populations = self.cache.load("pseudo_populations", alloc=self.natom, tags="o")[
                0
            ]
            self.logger.info("Computing atomic populations.")
            for index in range(self.natom):
                # pseudo_populations[i] = self.compute_pseudo_population(index)
                # Be consistent with do_spin_dens
                grid = self.get_grid(index)
                dens = self.get_moldens(index)
                at_weights = self.cache.load(f"at_weights_{index}")
                # weights correction
                wcor = self.get_wcor(index)
                # TODO: This is only for gLISA with grid_type == 1. We need to improve this.
                if at_weights.shape == self._grid.weights.shape:
                    at_weights = self.to_atomic_grid(index, at_weights)
                pseudo_populations[index] = grid.integrate(at_weights, dens * wcor)
            populations[:] = pseudo_populations
            populations += self.numbers - self.pseudo_numbers

    @just_once
    def do_charges(self):
        """Compute atomic charges."""
        # The charges are calculated in methods.
        charges, new = self._cache.load("charges", alloc=self.natom, tags="o")
        if new:
            self.do_populations()
            populations = self._cache.load("populations")
            self.logger.info("Computing atomic charges.")
            charges[:] = self.numbers - populations

    @just_once
    def do_spin_charges(self):
        """Compute atomic spin charges."""
        if self._spindens is not None:
            spin_charges, new = self._cache.load("spin_charges", alloc=self.natom, tags="o")
            self.do_partitioning()
            self.logger.info("Computing atomic spin charges.")
            for index in range(self.natom):
                grid = self.get_grid(index)
                spindens = self.get_spindens(index)
                at_weights = self.cache.load(f"at_weights_{index}")
                # weights correction
                wcor = self.get_wcor(index)
                spin_charges[index] = grid.integrate(at_weights, spindens * wcor)
                # spin_charges[index] = grid.integrate(at_weights, spindens)

    @just_once
    def do_moments(self):
        """
        Compute atomic multiple moments.

        Calculates various types of multipoles, including Cartesian, Spherical, and Radial moments.
        The order of the moments is determined by the `lmax` parameter.
        """
        ncart = get_ncart_cumul(self.lmax)
        cartesian_multipoles, new1 = self._cache.load(
            "cartesian_multipoles", alloc=(self.natom, ncart), tags="o"
        )

        npure = get_npure_cumul(self.lmax)
        pure_multipoles, new1 = self._cache.load(
            "pure_multipoles", alloc=(self.natom, npure), tags="o"
        )

        nrad = self.lmax + 1
        radial_moments, new2 = self._cache.load(
            "radial_moments", alloc=(self.natom, nrad), tags="o"
        )

        if new1 or new2:
            self.do_partitioning()
            print("Computing cartesian and pure AIM multipoles and radial AIM moments.")

            for i in range(self.natom):
                # 1) Define a 'window' of the integration grid for this atom
                center = self.coordinates[i]
                grid = self.get_grid(i)

                # 2) Compute the AIM
                at_weights = self.cache.load(f"at_weights_{i}")
                # TODO: This is only for gLISA with grid_type == 1. We need to improve this.
                if at_weights.shape == self._grid.weights.shape:
                    at_weights = self.to_atomic_grid(i, at_weights)
                aim = self.get_moldens(i) * at_weights

                # 3) Compute weight corrections
                wcor = self.get_wcor(i)

                # 4) Compute Cartesian multipole moments
                # The minus sign is present to account for the negative electron
                # charge.
                res = -grid.moments(
                    orders=self.lmax,
                    centers=np.asarray([center]),
                    func_vals=aim * wcor,
                    type_mom="cartesian",
                )
                cartesian_multipoles[i] = res.flatten()
                cartesian_multipoles[i, 0] += self.pseudo_numbers[i]

                # 5) Compute Pure multipole moments
                # The minus sign is present to account for the negative electron
                # charge.
                pure_multipoles[i] = -grid.moments(
                    orders=self.lmax,
                    centers=np.asarray([center]),
                    func_vals=aim * wcor,
                    type_mom="pure",
                ).flatten()
                pure_multipoles[i, 0] += self.pseudo_numbers[i]

                # 6) Compute Radial moments
                # For the radial moments, it is not common to put a minus sign
                # for the negative electron charge.
                radial_moments[i] = grid.moments(
                    orders=self.lmax,
                    centers=np.asarray([center]),
                    func_vals=aim * wcor,
                    type_mom="radial",
                ).flatten()

    def do_all(self):
        """Computes all properties and return a list of their keys."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr) and attr_name.startswith("do_") and attr_name != "do_all":
                attr()
        return list(self.cache.iterkeys(tags="o"))


class WPart(Part):
    """Base class for density partitioning schemes"""

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
        grid_type=1,
        density_cutoff=DENSITY_CUTOFF,
        negative_cutoff=NEGATIVE_CUTOFF,
        population_cutoff=POPULATION_CUTOFF,
        **kwargs,
    ):
        """
        Parameters
        ----------
        coordinates
             An array (N, 3) with centers for the atom-centered grids.
        numbers : 1D np.ndarray
             An array (N,) with atomic numbers.
        pseudo_numbers : 1D np.ndarray
             An array (N,) with effective charges. When set to None, this
             defaults to``numbers.astype(float)``.
        grid : Grid
             A Molecular integration grid. This must be a BeckeMolGrid
             instance with mode=='keep' or mode=='only'.
        moldens : 1D np.ndarray
             The spin-summed electron density on the grid.
        spindens : 1D np.ndarray or None, optional
             The spin difference density on the grid.
        local : bool, optional
             If ``True``: use the proper atomic grid for each AIM integral.
             If ``False``: use the entire molecular grid for each AIM integral.
        lmax : int, optional
             The maximum angular momentum in multipole expansions.
        """
        self._grid_type = grid_type
        self._on_molgrid = None
        self._only_use_molgrid = None
        self.setup_grids()
        local = not self._only_use_molgrid

        if local and grid.atgrids is None:
            raise ValueError(
                "Atomic grids are discarded from molecular grid object, "
                "but are needed for local integrations."
            )
        if logger is None:
            logger = logging.getLogger(self.__class__.__name__)
            setup_logger(logger)

        super().__init__(
            coordinates, numbers, pseudo_numbers, grid, moldens, spindens, local, lmax, logger
        )

        # Attributes for iterative density partitioning methods.
        self.history_propars = []
        self.history_charges = []
        self.history_entropies = []
        self.history_changes = []
        self.history_time_update_at_weights = []
        self.history_time_update_propars = []

        # attributes related to grids.
        self._radial_distances = []

        # Setup cutoff for numerical calculations.
        self._density_cutoff = density_cutoff
        self._population_cutoff = population_cutoff
        self._negative_cutoff = negative_cutoff

    def setup_grids(self):
        """Setup grids used in partitioning.

        # 1. atom_grids + mol_grid, use atoms for everything and mol_grid is only used for weights calculation
        # 2. atom_grids + mol_grid, use mol_grid for everything but atom_grids are used applied contratins.
        # 3. mol_grid, use mol_grid for everything, like in gLISA+.
        """
        assert self.grid_type in [1, 2, 3]
        self._on_molgrid = True if self.grid_type in [2, 3] else False
        self._only_use_molgrid = True if self.grid_type in [3] else False

    @property
    def grid_type(self):
        """Get the type of grids used for partitioning density.

        Returns
        -------
        str
            The type of grids used in the partitioning process.
        """
        return self._grid_type

    @property
    def only_use_molgrid(self):
        """
        Check whether intermediate values are computed exclusively using molecular grids.

        When set to `True`, all quantities are computed solely on the molecular grid.

        Returns
        -------
        bool
            True if intermediate values are computed only using the molecular grid, False otherwise.
        """
        return self._only_use_molgrid

    @property
    def on_molgrid(self):
        """
        Check whether quantities are computed on molecular grids.

        These grids are used for evaluating various properties, including:

        - AIM (Atoms-in-Molecule) weight functions: :math:`w_a(\\mathbf{r})`.
        - Pro-atom density: :math:`\rho_a^0(\\mathbf{r})`.
        - Pro-molecule density: :math:`\rho^0(\\mathbf{r})`.
        - Mean-square deviation during computations.

        Returns
        -------
        bool
            True if quantities are computed on molecular grids, False otherwise.
        """
        return self._on_molgrid

    @property
    def radial_distances(self):
        """
        Get the radial distances of points from the atomic coordinates.

        The radial distances are calculated as the L2 norm (Euclidean distance)
        of the points relative to the atomic coordinates.

        Notes
        -----
        Accessing this property triggers the calculation of radial distances
        via the `calc_radial_distances` method.

        Returns
        -------
        list
            A list containing the radial distances of points for each atom.
        """
        self.calc_radial_distances()
        return self._radial_distances

    @property
    def density_cutoff(self):
        """
        Get the density cutoff value.

        Density values below this cutoff are considered to be invalid.

        Returns
        -------
        float
            The cutoff value for density.
        """
        return self._density_cutoff

    @property
    def population_cutoff(self):
        """
        Get the population cutoff criterion.

        This represents the allowed difference between the sum of proatom parameters
        and the reference population for determining accuracy of methods.

        Returns
        -------
        float
            The cutoff value for population differences.
        """
        return self._population_cutoff

    @property
    def negative_cutoff(self):
        """
        Get the negative cutoff value.

        Values less than this threshold are treated as negative in computations.

        Returns
        -------
        float
            The negative cutoff value.
        """
        return self._negative_cutoff

    def _init_log_base(self):
        self.logger.info("Performing a density-based AIM analysis with a wavefunction as input.")
        deflist(
            self.logger,
            [
                ("Molecular grid", self._grid.__class__.__name__),
                ("Using local grids", self._local),
            ],
        )

    def _init_subgrids(self):
        # TODO: fix me, the subgrids are not necessary to be the atom grids.
        self._subgrids = self._grid.atgrids

    def to_atomic_grid(self, index, data):
        if index is None or not self.local:
            return data
        else:
            begin, end = self.grid.indices[index], self.grid.indices[index + 1]
            return data[begin:end]

    @just_once
    def calc_radial_distances(self):
        """Calculate radial distance w.r.t coordinates."""
        for iatom in range(self.natom):
            r = np.linalg.norm(self.grid.points - self.coordinates[iatom], axis=1)
            self._radial_distances.append(r)

    @just_once
    def do_density_decomposition(self):
        """Compute density decomposition."""
        if not self.local:
            self.logger.warning("Skip density decomposition because no local grids were found.")
            return

        for index in range(self.natom):
            atgrid = self.get_grid(index)
            assert isinstance(atgrid, AtomGrid)
            key = ("density_decomposition", index)
            if key not in self.cache:
                moldens = self.get_moldens(index)
                self.do_partitioning()
                print("Computing density decomposition for atom %i" % index)
                at_weights = self.cache.load(f"at_weights_{index}")
                # TODO: This is only for gLISA with grid_type == 1. We need to improve this.
                if at_weights.shape == self._grid.weights.shape:
                    at_weights = self.to_atomic_grid(index, at_weights)
                assert atgrid.l_max >= self.lmax
                splines = atgrid.radial_component_splines(moldens * at_weights)
                density_decomp = {"spline_%05i" % j: spl for j, spl in enumerate(splines)}
                self.cache.dump(key, density_decomp, tags="o")

    # @just_once
    # def do_hartree_decomposition(self):
    #     if not self.local:
    #         print(
    #             "!WARNING! Skip hartree decomposition because no local grids were found."
    #         )
    #         return

    #     for index in range(self.natom):
    #         key = ("hartree_decomposition", index)
    #         if key not in self.cache:
    #             self.do_density_decomposition()
    #             print("Computing hartree decomposition for atom %i" % index)
    #             density_decomposition = self.cache.load("density_decomposition", index)
    #             rho_splines = [
    #                 spline for foo, spline in sorted(density_decomposition.items())
    #             ]
    #             splines = solve_poisson_becke(rho_splines)
    #             hartree_decomp = dict(
    #                 ("spline_%05i" % j, spl) for j, spl in enumerate(splines)
    #             )
    #             self.cache.dump(key, hartree_decomp, tags="o")


def get_ncart_cumul(lmax):
    """The number of cartesian powers up to a given angular momentum, lmax."""
    return (lmax + 1) * (lmax + 2) * (lmax + 3) // 6


def get_npure_cumul(lmax):
    """The number of pure functions up to a given angular momentum, lmax."""
    return (lmax + 1) ** 2
