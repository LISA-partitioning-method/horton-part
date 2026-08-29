# HORTON-PART: molecular and periodic density partitioning.
# Copyright (C) 2023-2026 The HORTON-PART Development Team
#
# This file is part of HORTON-PART and is distributed under the GNU General
# Public License, version 3 or (at your option) any later version.
"""Command-line interface for periodic real-space density partitioning."""

from __future__ import annotations

import argparse

import numpy as np
from grid import PeriodicGrid

from horton_part.periodic import partition_periodic


def _load_input(filename):
    """Load and validate the array-level periodic input contract."""
    with np.load(filename, allow_pickle=False) as data:
        required = ("atcoords", "atnums", "points", "weights", "density")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Missing required input arrays: {', '.join(missing)}.")
        result = {name: np.asarray(data[name]) for name in required}
        result["cellvecs"] = (
            np.asarray(data["cellvecs"]) if "cellvecs" in data else np.zeros((0, 3))
        )
        result["pseudo_numbers"] = (
            np.asarray(data["atcorenums"])
            if "atcorenums" in data
            else result["atnums"].astype(float)
        )
        if "grid_sizes" in data:
            grid_sizes = np.asarray(data["grid_sizes"])
            if not np.issubdtype(grid_sizes.dtype, np.integer):
                raise ValueError("Grid sizes must use an integer dtype.")
            result["grid_sizes"] = grid_sizes.astype(int, copy=False)
        else:
            result["grid_sizes"] = None

    cellvecs = result["cellvecs"]
    if cellvecs.ndim != 2 or cellvecs.shape[1:] != (3,) or len(cellvecs) > 3:
        raise ValueError("Cell vectors must have shape (nperiodic, 3), with 0 <= nperiodic <= 3.")
    grid_sizes = result["grid_sizes"]
    if grid_sizes is not None:
        if (
            grid_sizes.ndim != 1
            or not len(grid_sizes)
            or np.any(grid_sizes <= 0)
            or grid_sizes.sum() != len(result["points"])
        ):
            raise ValueError(
                "Grid sizes must be positive block lengths whose sum equals the number of points."
            )
    return result


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="part-periodic",
        description="Partition a molecular or periodic density supplied on a real-space grid.",
    )
    parser.add_argument("input", help="NPZ file containing coordinates, grid, and density arrays.")
    parser.add_argument("output", help="Output NPZ file for charges and optional AIM weights.")
    parser.add_argument(
        "--method",
        default="mbis",
        choices=("hirshfeld", "hirshfeld-i", "mbis", "lisa", "avh"),
        help="Partitioning method (default: mbis).",
    )
    parser.add_argument(
        "--basis",
        help="LISA or radial-spline pro-atom JSON file; required for spline methods.",
    )
    parser.add_argument(
        "--avh-variant",
        default="B",
        choices=("A", "B", "M", "a", "b", "m"),
        help="AVH state selection: all, physically bound, or minimal neutral (default: B).",
    )
    parser.add_argument("--threshold", type=float, default=1.0e-7)
    parser.add_argument("--inner-threshold", type=float, default=1.0e-9)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--inner-maxiter", type=int, default=1000)
    parser.add_argument("--density-cutoff", type=float, default=1.0e-10)
    parser.add_argument("--mixing", type=float, default=0.5)
    parser.add_argument(
        "--no-aim-weights",
        action="store_true",
        help="Do not store the potentially large per-atom weight array.",
    )
    return parser


def main(argv=None):
    """Run periodic partitioning from an NPZ input file."""
    args = _build_parser().parse_args(argv)
    data = _load_input(args.input)
    coordinates = data["atcoords"]
    numbers = data["atnums"]
    points = data["points"]
    weights = data["weights"]
    density = data["density"]
    cellvecs = data["cellvecs"]
    pseudo_numbers = data["pseudo_numbers"]

    grid = PeriodicGrid(points, weights, cellvecs, wrap=bool(cellvecs.size))
    result = partition_periodic(
        args.method,
        coordinates,
        numbers,
        grid,
        density,
        basis=args.basis,
        pseudo_numbers=pseudo_numbers,
        threshold=args.threshold,
        inner_threshold=args.inner_threshold,
        maxiter=args.maxiter,
        inner_maxiter=args.inner_maxiter,
        density_cutoff=args.density_cutoff,
        mixing=args.mixing,
        avh_variant=args.avh_variant,
        return_weights=not args.no_aim_weights,
    )
    density_integral = float(np.dot(weights, density))
    output = result.to_dict()
    output.update(
        {
            "atcoords": coordinates,
            "atnums": numbers,
            "atcorenums": pseudo_numbers,
            "cellvecs": cellvecs,
            "density_integral": np.array(density_integral),
            "total_charge": np.array(pseudo_numbers.sum() - density_integral),
            "charge_conservation_error": np.array(
                result.charges.sum() - (pseudo_numbers.sum() - density_integral)
            ),
        }
    )
    if data["grid_sizes"] is not None:
        output["grid_sizes"] = data["grid_sizes"]
    np.savez_compressed(args.output, **output)
    return 0


if __name__ == "__main__":
    main()
