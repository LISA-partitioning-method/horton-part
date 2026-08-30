Periodic real-space partitioning
================================

HORTON-Part can partition densities on arbitrary ``qc-grid``
``PeriodicGrid`` objects. Periodic images are evaluated through local translated
grids, so atoms near a cell face contribute correctly at the opposite face without
replicating the full unit cell. The initial periodic API supports Hirshfeld,
Hirshfeld-I, MBIS, LISA, and AVH-A/B/M. An explicitly named
``avh-supplied`` mode is also available for reproducing calculations made
with a user-defined subset of charged states.

Start with the executable :doc:`periodic quick-start notebook
<notebooks/periodic_quick_start>`, then use the :doc:`five-method comparison
<notebooks/periodic_methods>` to review basis requirements and numerical checks.
Both examples construct a small synthetic cell and require neither GPAW nor external data.

Generating an input archive from GPAW
-------------------------------------

``part-from-gpaw`` converts a legacy GPAW restart into the all-electron NPZ
representation consumed by ``part-periodic``. Run it inside the environment where
GPAW is already installed; GPAW is deliberately not a HORTON-Part dependency.
The converter currently requires GPAW legacy mode and one process because it reads
the calculator's PAW setup and atomic density-matrix internals.

.. code-block:: bash

   part-from-gpaw calculation.gpw density.npz
   part-periodic density.npz mbis.npz --method mbis

The archive combines the uniform pseudo-density grid with one atom-centered PAW
augmentation-correction grid per atom. It records the block lengths in
``grid_sizes`` and writes coordinates, cell vectors, weights, and densities in
atomic units. This is the same complete representation used for the periodic
partitioning calculations; a pseudo-density-only archive is not sufficient.

Command-line use
----------------

``part-periodic`` reads an NPZ file containing ``atcoords`` (bohr), ``atnums``,
``points`` (bohr), ``weights`` (bohr cubed), and ``density`` (electrons per bohr
cubed). ``cellvecs`` contains zero to three lattice vectors in bohr; omit it for a
finite grid. An optional ``atcorenums`` array defines the charge reference used for
each atom. All arrays must use one consistent all-electron convention.

DensPart GPAW archives may concatenate a periodic uniform block and atom-centered
PAW augmentation blocks. Their optional ``grid_sizes`` array gives the positive
length of each block and must sum to the total point count. HORTON-Part integrates
the concatenated quadrature directly and preserves ``grid_sizes`` in its output.
Zero-weight points and negative density noise within HORTON-Part's numerical
tolerance are accepted. The current spline-state format stores densities that
integrate to ``Z - charge``; valence-only spline libraries are not yet supported.

.. code-block:: bash

   part-periodic density.npz mbis.npz --method mbis
   part-periodic density.npz hi.npz --method hirshfeld-i \
       --basis pbe-6311pgdp-sapporo-dkh3tzp-dkh2.json
   part-periodic density.npz avh.npz --method avh \
       --basis pbe-6311pgdp-sapporo-dkh3tzp-dkh2.json --avh-variant B
   part-periodic density.npz avh-supplied.npz --method avh \
       --basis custom-states.json --avh-variant supplied

LISA, AVH, and MBIS accept ``--solver optimizer`` (the default) or
``--solver sc``. The SC route uses the same nonnegative fixed-point coefficient
updates as the molecular implementation; MBIS additionally updates every shell
exponent from its first radial moment. It does not call ``scipy.optimize.minimize``.
For AVH, all selected states start with coefficient one so that multiplicative SC
updates can activate them. Solver performance is system dependent, so compare both
routes before selecting one for production. Nearly linearly dependent LISA or AVH
bases can make plain SC converge very slowly; the optimizer remains the recommended
default for such cases:

.. code-block:: bash

   part-periodic density.npz lisa-sc.npz --method lisa \
       --basis lisa.json --solver sc
   part-periodic density.npz mbis-sc.npz --method mbis --solver sc

The output ``solver`` field records the selected route. Plain Hirshfeld has no
optimized coefficients, while Hirshfeld-I already uses its own charge-interpolation
fixed-point cycle; therefore ``--solver`` does not apply to those methods.

The radial-spline file used by Hirshfeld, Hirshfeld-I, and AVH follows the shared
``aim-proatom-spline-v1`` schema. LISA accepts the bundled HORTON-Part basis,
the legacy three-array mapping, or ``aim-lisa-basis-v1``. The former ``denspart-*``
schema identifiers remain readable for compatibility.

The package-neutral LISA file is accepted by both finite and periodic LISA. Likewise,
the spline file can initialize finite Hirshfeld/Hirshfeld-I and all periodic spline
methods, so a published atomic library does not need consumer-specific conversion.

The same spline file can construct a finite-system reference database:

.. code-block:: python

   from horton_part import ProAtomDB

   proatomdb = ProAtomDB.from_spline_file("pbe-6311pgdp-molecular.json")

The output stores charges, populations, pro-atom parameters, the promolecular
density, convergence history, and charge-conservation diagnostics. ``charges`` and
``populations`` are always obtained by integrating the final AIM weights on the input
quadrature. ``model_charges`` and ``model_populations`` separately expose the
analytic populations of the fitted pro-atoms; small differences diagnose grid and
cutoff errors. Per-atom AIM weights are included by default; use
``--no-aim-weights`` when only charges are needed. In that mode,
``reconstruction_error`` is not evaluated and is stored as NaN.

Python API
----------

.. code-block:: python

   from horton_part.periodic import partition_periodic

   result = partition_periodic(
       "mbis", coordinates, numbers, grid, density, return_weights=True
   )
   print(result.charges)

``partition_periodic`` also accepts ``basis``, ``pseudo_numbers``, iteration
thresholds, and the AVH variant. When requested, the returned ``aim_weights`` array
has shape ``(natom, npoint)`` and sums to one wherever the promolecular density is
nonzero. Dense per-atom arrays are not constructed when ``return_weights=False``.

Real-material parity validation
-------------------------------

An optional slow test compares all five methods with archived DensPart GaP weights.
Configure the paths to an all-electron GaP archive and its basis libraries, then run:

.. code-block:: bash

   export HORTON_PART_GAP_ROOT=/path/to/gap/production
   export HORTON_PART_GAP_LISA_BASIS=/path/to/lisa.json
   export HORTON_PART_GAP_SPLINE_ALL_BASIS=/path/to/spline-all.json
   export HORTON_PART_GAP_SPLINE_BOUND_BASIS=/path/to/spline-bound.json
   pytest --slow tests/test_periodic_gap.py

The test integrates the independently archived DensPart weights rather than comparing
printed pro-atom populations. This distinction is important when finite-grid or radial
cutoff errors make the two charge definitions differ slightly. Method-specific tolerances
are at most ``5e-4`` electrons to accommodate optimizer stopping-point differences across
SciPy versions; deterministic spline cases use tighter tolerances. The widest tolerance is
for KBBF LISA, whose 145-parameter objective is particularly flat near the minimum.

The corresponding nine-atom KBBF test exercises both the memory-saving local-only path
and full AIM-weight reconstruction:

.. code-block:: bash

   export HORTON_PART_KBBF_ROOT=/path/to/kbbf/partition/archive
   export HORTON_PART_KBBF_MBIS_REFERENCE=/path/to/kbbf/mbis.npz
   export HORTON_PART_KBBF_LISA_BASIS=/path/to/lisa.json
   export HORTON_PART_KBBF_SPLINE_ALL_BASIS=/path/to/spline-all.json
   export HORTON_PART_KBBF_SPLINE_BOUND_BASIS=/path/to/spline-bound.json
   sbatch tests/run_kbbf_parity.slurm

These real-material tests are optional because neither the density archives nor the
independently generated reference weights are distributed with HORTON-PART.

Method scope
------------

Hirshfeld uses the neutral spline state without optimization. Hirshfeld-I mixes
adjacent integer-charge states, including an implicit zero-density fully stripped
endpoint. Its active integer-charge spline states are cached on demand, and only their
mixing coefficients change during iteration. LISA and AVH optimize nonnegative mixing
coefficients directly from cached local basis blocks; AVH-A retains all available
charged states, AVH-B retains physically bound states, and AVH-M retains only the
neutral state. MBIS globally optimizes its minimal Slater shells on cached shell-local
grids, expanding a grid only when its optimized shell becomes more diffuse. Orbital-matrix
methods such as Mulliken partitioning are outside the real-space grid API.

AVH validates the state set before optimization. AVH-A requires every charge from
``-3`` through ``Z-1``; AVH-B requires ``0`` through ``Z-1`` and includes the
monoanion only when it is marked as bound. This prevents a short Hirshfeld-I state
library from being mistaken for a complete AVH basis.

``avh-supplied`` deliberately skips the A/B/M completeness contract and optimizes
every populated state present in the input basis. Its result is labelled
``avh-supplied`` rather than AVH-A or AVH-B. This mode is intended for transparent
reproduction of legacy or deliberately truncated state libraries, not as an alias
for a formal AVH variant.
