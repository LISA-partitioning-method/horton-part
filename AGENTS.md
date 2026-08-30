# Repository Guidelines

## Project Structure & Module Organization

HORTON-Part uses a `src/` layout. Finite-system methods and shared infrastructure live in
`src/horton_part/`; periodic-grid implementations are isolated in `src/horton_part/periodic/`.
Command-line programs and density converters are in `src/horton_part/scripts/`, while bundled
basis data and YAML defaults are in `src/horton_part/data/`. Tests are under `tests/`, including
focused periodic tests and optional real-data parity tests. User documentation is in `docs/`,
with executable examples in `docs/notebooks/`. Record user-visible changes in `CHANGELOG.md`.

## Build, Test & Documentation Commands

Use Python 3.10 or newer:

```bash
python -m pip install -e '.[dev,tests]'
pytest -q
pytest tests/test_periodic.py -q
ruff check src/ tests/
pre-commit run --all-files
cd docs && make html
```

Tests marked `slow` require external article data and are not part of routine local validation.
When changing notebooks, execute them with `jupyter nbconvert --execute` and confirm the Sphinx
build. Do not commit notebook outputs, build directories, caches, or generated egg-info files.

## Coding Style & API Boundaries

Use four-space indentation, a 100-character line limit, `snake_case` functions, and
`CapWords` classes. Preserve the common result fields and method dispatch exposed by the periodic
API and CLI. Finite and periodic loaders should consume the same package-neutral `aim-*` basis
schemas while retaining documented legacy formats. Keep GPAW optional: converters may import it
lazily, but the package must install and run without GPAW. Validate charge conservation, density
reconstruction, partition of unity, state completeness, and solver convergence explicitly.
Retain GPL notices and document copied or adapted source in `NOTICE`.

## Testing, Commits & Pull Requests

Add unit tests beside the affected subsystem and parity tests when finite and periodic paths
should agree. Use defensible numerical tolerances and test both optimizer and self-consistent
routes when changing coefficient updates. Follow the existing short imperative commit style,
for example `optimize periodic methods on sparse local grids`. Pull requests should describe
public API and numerical effects, list validation commands, identify optional dependencies, and
include documentation updates for new methods, inputs, or result fields.
