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
    with np.load(args.input, allow_pickle=False) as data:
        required = ("atcoords", "atnums", "points", "weights", "density")
        missing = [name for name in required if name not in data]
        if missing:
            raise ValueError(f"Missing required input arrays: {', '.join(missing)}.")
        coordinates = np.asarray(data["atcoords"])
        numbers = np.asarray(data["atnums"])
        points = np.asarray(data["points"])
        weights = np.asarray(data["weights"])
        density = np.asarray(data["density"])
        cellvecs = np.asarray(data["cellvecs"]) if "cellvecs" in data else np.zeros((0, 3))
        pseudo_numbers = (
            np.asarray(data["atcorenums"]) if "atcorenums" in data else numbers.astype(float)
        )

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
    np.savez_compressed(args.output, **output)
    return 0


if __name__ == "__main__":
    main()
