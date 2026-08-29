Periodic real-space partitioning
================================

HORTON-Part can partition densities on arbitrary ``qc-grid``
``PeriodicGrid`` objects. Periodic images are evaluated through local translated
grids, so atoms near a cell face contribute correctly at the opposite face without
replicating the full unit cell. The initial periodic API supports Hirshfeld,
Hirshfeld-I, MBIS, LISA, and AVH-A/B/M.

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
       --basis pbe-periodic.json
   part-periodic density.npz avh.npz --method avh \
       --basis pbe-periodic.json --avh-variant B

The radial-spline file used by Hirshfeld, Hirshfeld-I, and AVH must follow
``denspart-spline-proatom-basis-v1``. LISA accepts the bundled HORTON-Part basis,
the legacy three-array mapping, or ``denspart-lisa-basis-v1``.

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
are at most ``1e-4`` electrons to accommodate optimizer stopping-point differences across
SciPy versions; deterministic spline cases use tighter tolerances.

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
