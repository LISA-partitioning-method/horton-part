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
"""Base classes for (atoms-in-molecules) partitioning algorithms"""


import numpy as np

from .cache import JustOnceClass, just_once, Cache
from .utils import typecheck_geo
from .log import log
from grid import AtomGrid


__all__ = ["Part", "WPart"]


class Part(JustOnceClass):
    name = None
    linear = False  # whether the populations are linear in the density matrix.

    def __init__(
        self, coordinates, numbers, pseudo_numbers, grid, moldens, spindens, local, lmax
    ):
        """
        **Arguments:**

        coordinates
             An array (N, 3) with centers for the atom-centered grids.

        numbers
             An array (N,) with atomic numbers.

        pseudo_numbers
             An array (N,) with effective charges. When set to None, this
             defaults to``numbers.astype(float)``.

        grid
             The integration grid

        moldens
             The spin-summed electron density on the grid.

        spindens
             The spin difference density on the grid. (Can be None)

        local
             Whether or not to use local (non-periodic) subgrids for atomic
             integrals.

        lmax
             The maximum angular momentum in multipole expansions.
        """
        # Init base class
        JustOnceClass.__init__(self)

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

        # Initialize the subgrids
        if local:
            self._init_subgrids()
        # Some screen logging
        self.biblio = []
        self._init_log_base()
        self._init_log_scheme()
        self._init_log_memory()

    def __getitem__(self, key):
        return self.cache.load(key)

    @property
    def natom(self):
        return self._natom

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def numbers(self):
        return self._numbers

    @property
    def pseudo_numbers(self):
        return self._pseudo_numbers

    @property
    def grid(self):
        return self.get_grid()

    @property
    def local(self):
        return self._local

    @property
    def lmax(self):
        return self._lmax

    @property
    def cache(self):
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
        result = self.to_atomic_grid(index, self._moldens)
        if output is not None:
            output[:] = result
        return result

    def get_spindens(self, index=None, output=None):
        result = self.to_atomic_grid(index, self._spindens)
        if output is not None:
            output[:] = result
        return result

    def get_wcor(self, index):
        """Return the weight corrections on a grid

        See get_grid for the meaning of the optional arguments
        """
        raise NotImplementedError

    def _init_subgrids(self):
        raise NotImplementedError

    def _init_log_base(self):
        raise NotImplementedError

    def _init_log_scheme(self):
        raise NotImplementedError

    def _init_log_memory(self):
        if log.do_medium:
            # precompute arrays sizes for certain grids
            nbyte_global = self.grid.size * 8
            nbyte_locals = np.array(
                [self.get_grid(i).size * 8 for i in range(self.natom)]
            )

            # compute and report usage
            estimates = self.get_memory_estimates()
            nbyte_total = 0
            print("Coarse estimate of memory usage for the partitioning:")
            print("                         Label  Memory[GB]")
            print()
            for label, nlocals, nglobal in estimates:
                nbyte = np.dot(nlocals, nbyte_locals) + nglobal * nbyte_global
                print("%30s  %10.3f" % (label, nbyte / 1024.0**3))
                nbyte_total += nbyte
            print("%30s  %10.3f" % ("Total", nbyte_total / 1024.0**3))
            print("" + "~" * 100)
            print()

    def get_memory_estimates(self):
        return [
            ("Atomic weights", np.ones(self.natom), 0),
            ("Promolecule", np.zeros(self.natom), 1),
            ("Working arrays", np.zeros(self.natom), 2),
        ]

    def to_atomic_grid(self, index, data):
        raise NotImplementedError

    def compute_pseudo_population(self, index):
        grid = self.get_grid(index)
        dens = self.get_moldens(index)
        at_weights = self.cache.load("at_weights", index)
        wcor = self.get_wcor(index)
        wcor = np.ones_like(at_weights) if wcor is None else wcor
        return grid.integrate(at_weights, dens, wcor)

    @just_once
    def do_partitioning(self):
        self.update_at_weights()

    do_partitioning.names = []

    def update_at_weights(self):
        """Updates the at_weights arrays in the case (and all related arrays)"""
        raise NotImplementedError

    @just_once
    def do_populations(self):
        populations, new = self.cache.load("populations", alloc=self.natom, tags="o")
        if new:
            self.do_partitioning()
            pseudo_populations = self.cache.load(
                "pseudo_populations", alloc=self.natom, tags="o"
            )[0]
            print("Computing atomic populations.")
            for i in range(self.natom):
                pseudo_populations[i] = self.compute_pseudo_population(i)
            populations[:] = pseudo_populations
            populations += self.numbers - self.pseudo_numbers

    @just_once
    def do_charges(self):
        charges, new = self._cache.load("charges", alloc=self.natom, tags="o")
        if new:
            self.do_populations()
            populations = self._cache.load("populations")
            print("Computing atomic charges.")
            charges[:] = self.numbers - populations

    @just_once
    def do_spin_charges(self):
        if self._spindens is not None:
            spin_charges, new = self._cache.load(
                "spin_charges", alloc=self.natom, tags="o"
            )
            self.do_partitioning()
            print("Computing atomic spin charges.")
            for index in range(self.natom):
                grid = self.get_grid(index)
                spindens = self.get_spindens(index)
                at_weights = self.cache.load("at_weights", index)
                wcor = self.get_wcor(index)
                spin_charges[index] = grid.integrate(at_weights, spindens, wcor)

    @just_once
    def do_moments(self):
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
                aim = self.get_moldens(i) * self.cache.load("at_weights", i)

                # 3) Compute weight corrections
                wcor = self.get_wcor(i)
                wcor = 1 if wcor is None else wcor

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
        local=True,
        lmax=3,
    ):
        """
        **Arguments:**

        coordinates
             An array (N, 3) with centers for the atom-centered grids.

        numbers
             An array (N,) with atomic numbers.

        pseudo_numbers
             An array (N,) with effective charges. When set to None, this
             defaults to``numbers.astype(float)``.

        grid
             A Molecular integration grid. This must be a BeckeMolGrid
             instance with mode=='keep' or mode=='only'.

        moldens
             The spin-summed electron density on the grid.

        **Optional arguments:**

        spindens
             The spin difference density on the grid.

        local
             If ``True``: use the proper atomic grid for each AIM integral.
             If ``False``: use the entire molecular grid for each AIM integral.

        lmax
             The maximum angular momentum in multipole expansions.
        """
        if local and grid.atgrids is None:
            raise ValueError(
                "Atomic grids are discarded from molecular grid object, "
                "but are needed for local integrations."
            )
        Part.__init__(
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

    def _init_log_base(self):
        if log.do_medium:
            log("Performing a density-based AIM analysis with a wavefunction as input.")
            log.deflist(
                [
                    (" Molecular grid", self._grid.__class__.__name__),
                    (" Using local grids", self._local),
                ]
            )

    def _init_subgrids(self):
        self._subgrids = self._grid.atgrids

    def get_wcor(self, index):
        return None

    def to_atomic_grid(self, index, data):
        if index is None or not self.local:
            return data
        else:
            begin, end = self.grid.indices[index], self.grid.indices[index + 1]
            return data[begin:end]

    @just_once
    def do_density_decomposition(self):
        if not self.local:
            print(
                "!WARNING! Skip density decomposition because no local grids were found."
            )
            return

        for index in range(self.natom):
            atgrid = self.get_grid(index)
            assert isinstance(atgrid, AtomGrid)
            key = ("density_decomposition", index)
            if key not in self.cache:
                moldens = self.get_moldens(index)
                self.do_partitioning()
                print("Computing density decomposition for atom %i" % index)
                at_weights = self.cache.load("at_weights", index)
                assert atgrid.l_max >= self.lmax
                splines = atgrid.radial_component_splines(moldens * at_weights)
                density_decomp = dict(
                    ("spline_%05i" % j, spl) for j, spl in enumerate(splines)
                )
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
