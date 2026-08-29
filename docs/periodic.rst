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
cubed). ``cellvecs`` contains zero to three lattice vectors; omit it for a finite
grid. An optional ``atcorenums`` array defines the reference nuclear charges.

.. code-block:: bash

   part-periodic density.npz mbis.npz --method mbis
   part-periodic density.npz hi.npz --method hirshfeld-i \
       --basis pbe-periodic.json
   part-periodic density.npz avh.npz --method avh \
       --basis pbe-periodic.json --avh-variant B

The radial-spline file used by Hirshfeld, Hirshfeld-I, and AVH must follow
``denspart-spline-proatom-basis-v1``. LISA accepts the bundled HORTON-Part basis,
the legacy three-array mapping, or ``denspart-lisa-basis-v1``. The pro-atom data
and input density must use the same all-electron or valence-electron convention.

The output stores charges, populations, pro-atom parameters, the promolecular
density, convergence history, and charge-conservation diagnostics.
``grid_populations`` separately records the numerical integral of each stockholder
density, which is useful for diagnosing finite-grid and cutoff errors. Per-atom AIM
weights are included by default; use ``--no-aim-weights`` when only charges are needed.

Python API
----------

.. code-block:: python

   from horton_part.periodic import partition_periodic

   result = partition_periodic(
       "mbis", coordinates, numbers, grid, density, return_weights=True
   )
   print(result.charges)

``partition_periodic`` also accepts ``basis``, ``pseudo_numbers``, iteration
thresholds, and the AVH variant. The returned ``aim_weights`` array has shape
``(natom, npoint)`` and sums to one wherever the promolecular density is nonzero.

Method scope
------------

Hirshfeld uses the neutral spline state without optimization. Hirshfeld-I mixes
adjacent integer-charge states, including an implicit zero-density fully stripped
endpoint. LISA and AVH optimize nonnegative mixing coefficients; AVH-A retains all
available charged states, AVH-B retains physically bound states, and AVH-M retains
only the neutral state. MBIS optimizes its minimal Slater shells. Orbital-matrix
methods such as Mulliken partitioning are outside the real-space grid API.

AVH validates the state set before optimization. AVH-A requires every charge from
``-3`` through ``Z-1``; AVH-B requires ``0`` through ``Z-1`` and includes the
monoanion only when it is marked as bound. This prevents a short Hirshfeld-I state
library from being mistaken for a complete AVH basis.
