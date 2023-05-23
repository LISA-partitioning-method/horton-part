# HORTON-PART
<a href='https://docs.python.org/3.10/'><img src='https://img.shields.io/badge/python-3.10-blue.svg'></a>


## About
This package implements various partitioning schemes described in<a href=https://doi.org/10.1063/5.0076630>This paper</a>.

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

DensPart is distributed under GPL License version 3 (GPLv3).


## Dependencies

The following dependencies will be necessary for DensPart to build properly,

* Python >= 3.10: http://www.python.org/
* SciPy : http://www.scipy.org/
* NumPy : http://www.numpy.org/
* Nosetests : http://readthedocs.org/docs/nose/en/latest/
* horton-grid : http://github.com/yingxingcheng/horton-grid

The dependence on `horton-grid` is only because of the `grid` module. This will be replaced by `qcgrids` in
the near future.


## Installation

To install DensPart:

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
