# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.4]  - 2024-05-16

### Added
- Add script to generate cube file to visualize results.

## [1.1.3]  - 2024-04-08

### Added
- Fix the entropy of the first iteration of LISA/gLISA solvers.
- Fix docs

## [1.1.2]  - 2024-03-24

### Added
- Package version in documentation.

### Fixed
- Initial values for GLISA-SC/LISA-SC and Newton methods.

## [1.1.1]  - 2024-03-16

### Added
- Update new logo

### Fixed
- GISA docs error.

## [1.1.0]  - 2024-01-17

### Added

- Documentation
- CDIIS solver instead of using the old DIIS solver in LISA scheme.
- Global LISA method with the same sovlers used in local LISA scheme.
- Unit tests
- Main program with support for a `yaml` configure file.
- General exponential basis functions.
- Big rewrite with new API.

### Fixed

- DIIS solver

## [1.0.10] - 2023-08-29

### Added

- Different solvers for LISA model using `trust_constr`

## [1.0.9] - 2023-08-26

### Added

- Time usae for each process during partitioning
- Cache file for density and grids.

## [1.0.8] - 2023-08-17

### Added

- Add `LISA-12` model.
- Add comparision for `LISA-1` and `LSIA-2`.

## [1.0.7] - 2023-08-15

### Fixed

- Fix the Lagrange method of GISA and LISA, and use the difference between densities.

## [1.0.6] - 2023-08-15

### Added

- Gaussian basis sets for F, Si, S, Br atoms.

## [1.0.5] - 2023-08-15

### Fixed

- The version info in log information.

## [1.0.4] - 2023-08-15

### Added

- Add `part` script to run `lisa`, `gisa`, `mbis` and `isa` methods.

## [Unreleased]

## [1.0.3] - 2023-06-29

### Added

- Add `gbasis` dependency for AIM moment calculation
- Add `moments` module.

## [Unreleased]

## [1.0.2] - 2023-06-19

### Fixed

- Project dependencies
- Fix a bug of ISA method, in which the $4 \times \pi * r**2$ is missed when doing radial integrating
- Replace `nose` dependencies with `pytest` and `pytest-skip-slow`

## [1.0.1] - 2023-06-18

### Fixed

- Fix examples by removing `horton-grid` dependency

## [1.0.0] - 2023-06-18

### Added

- Support new [grid](https://github.com/theochem/grid) and remove `horton-grid` support

## [0.0.2] - 2023-05-23

### Fixed

- Upload to PyPI

## [0.0.1] - 2023-05-23

### Added

- Examples for ISA methods
- Tests for all methods
- L-ISA method using two different optimization approaches
- G-ISA method
- The Python 3.10 support
