# HORTON-PART
<a href='https://docs.python.org/3.10/'><img src='https://img.shields.io/badge/python-3.10-blue.svg'></a>

```text
P_A__R_T
/ (..) \  Welcome to HORTON-PART!
\/ || \/
 |_''_|
```
[HORTON-PART](https://github.com/yingxingcheng/horton-part) is a computational chemistry package that supports different partition schemes.It is based on the sub-module `part` of `HORTON2`, which is written and maintained by Toon Verstraelen (1).

In HORTON3, all sub-modules have been rewritten using the pure Python programming language to support Python 3+. The 'part' module has also been rewritten and is now called [denspart](https://github.com/theochem/denspart) module.

However, the algorithm implemented in `denspart` only uses one-step optimization, which can be computationally expensive for large systems. Additionally, `denspart` only supports the `MBIS` partitioning scheme.

Another `part` module has been rewritten in pure Python programming language by Farnaz Heidar-Zadeh (2). However, the integration grid implemented in this module still uses the old 'grid' from Horton2. [HORTON-PART](https://github.com/yingxingcheng/horton-part) with version `0.0.X` is based on this module. Starting from version `1.X.X`, [HORTON-PART](https://github.com/yingxingcheng/horton-part) supports the new integration [qc-grid](https://github.com/theochem/grid).

This version contains contributions from YingXing Cheng (1), Toon Verstraelen (1), Pawel Tecmer (2), Farnaz Heidar-Zadeh (2), Cristina E. González-Espinoza (2), Matthew Chan (2), Taewon D. Kim (2), Katharina Boguslawski (2), Stijn Fias (3), Steven Vandenbrande (1), Diego Berrocal (2), and Paul W. Ayers (2)

- (1) Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium.
- (2) The Ayers Group, McMaster University, Hamilton, Ontario, Canada.
- (3) General Chemistry (ALGC), Free University of Brussels, Brussels, Belgium.

More information about HORTON can be found on this [website](http://theochem.github.com/horton/).


## About
This package implements various partitioning schemes described in <a href=https://doi.org/10.1063/5.0076630>this paper</a>, including:

- Becke method
- Mulliken method
- Hirshfeld partitioning scheme
- Iterative Hirshfeld (Hirshfeld-I) partitioning scheme
- Iterative stockholder approach (ISA)
- Gaussian iterative stockholder approach (GISA)
- Minimal Basis Iterative Stockholder (MBIS)
- Linear approximation of the ISA (L-ISA) method

## License

`horton-part` is distributed under GPL License version 3 (GPLv3).


## Dependencies

The following dependencies will be necessary for `horton-part` to build properly,

* SciPy : http://www.scipy.org/
* NumPy : http://www.numpy.org/
* quadprog>=0.1.11 : https://github.com/quadprog/quadprog
* cvxopt>=1.3.1 : https://github.com/cvxopt/cvxopt
* qc-grid : https://github.com/theochem/grid
* qc-iodata : https://github.com/theochem/iodata
* gbasis : https://github.com/theochem/gbasis


## Installation

To install `horton-part` with version `0.0.x`:

```bash
pip install horton-part
```

To install latest `horton-part`:

```bash
git clone http://github.com/yingxingcheng/horton-part
cd horton-part
pip install .
```

To run test, one needs to add tests dependencies for `tests`:

```bash
pip install .[tests]
```

For developers, one could need all dependencies:
```bash
pip install -e .[dev,tests]
```
