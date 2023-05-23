# HORTON-PART
<a href='https://docs.python.org/3.10/'><img src='https://img.shields.io/badge/python-3.10-blue.svg'></a>


## About
This package implements various partitioning schemes described in <a href=https://doi.org/10.1063/5.0076630>this paper</a>.

The methods included in this library are:

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

* Python >= 3.10: http://www.python.org/
* SciPy : http://www.scipy.org/
* NumPy : http://www.numpy.org/
* Nosetests : http://readthedocs.org/docs/nose/en/latest/
* pytest : https://docs.pytest.org/en/7.3.x/contents.html
* matplotlib : https://matplotlib.org/
* quadprog>=0.1.11 : https://github.com/quadprog/quadprog
* cvxopt>=1.3.1 : https://github.com/cvxopt/cvxopt
* qc-iodata : https://github.com/theochem/iodata
* horton-grid : https://github.com/yingxingcheng/horton-grid
* pep517 : https://peps.python.org/pep-0517/ (for developers)
* pre-commit : https://pre-commit.com/ (for developers)
* horton-grid : http://github.com/yingxingcheng/horton-grid

In order to use horton-grid, the following libraries with latest version should be installed manually.

* gbasis : https://github.com/theochem/gbasis (for running examples only)

To fix the error and install the required libraries, follow these steps:

- Go to the gbasis GitHub repository: https://github.com/theochem/gbasis
- Follow the installation instructions provided in the repository's README file.

The dependence on `horton-grid` is only because of the `grid` module. This will be replaced by `qcgrids` in
the near future.


## Installation

To install `horton-part`:

```bash
git clone http://github.com/yingxingcheng/horton-part
cd horton-part
pip install . [--user]
```

For developers:
```bash
pip install -e .
```


## Testing

To run tests:

```bash
git clone http://github.com/yingxingcheng/horton-part
pytest horton-part
```
