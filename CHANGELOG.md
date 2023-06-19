# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
