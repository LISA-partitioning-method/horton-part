#!/bin/bash
#SBATCH --job-name=horton-gap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=amd192c
#SBATCH --time=01:00:00
#SBATCH --mem=16G

set -euo pipefail

repo_dir=${HORTON_PART_REPO_DIR:-${SLURM_SUBMIT_DIR:-}}
if [[ -z "$repo_dir" ]]; then
    echo "Set HORTON_PART_REPO_DIR when SLURM_SUBMIT_DIR is unavailable." >&2
    exit 2
fi
repo_dir=$(cd -- "$repo_dir" && pwd)
python=${HORTON_PART_TEST_PYTHON:-python}

required=(
    HORTON_PART_GAP_ROOT
    HORTON_PART_GAP_LISA_BASIS
    HORTON_PART_GAP_SPLINE_ALL_BASIS
    HORTON_PART_GAP_SPLINE_BOUND_BASIS
)
for variable in "${required[@]}"; do
    [[ -n "${!variable:-}" ]] || {
        echo "Missing required environment variable: ${variable}" >&2
        exit 2
    }
done

cd "$repo_dir"
export PYTHONPATH="${repo_dir}/src${PYTHONPATH:+:${PYTHONPATH}}"
pytest_options=(-v)
if "$python" -m pytest --help | grep -q -- "--slow"; then
    pytest_options+=(--slow)
fi
"$python" -m pytest "${pytest_options[@]}" "$@" tests/test_periodic_gap.py
