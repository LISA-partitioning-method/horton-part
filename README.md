DensPart
======
<a href='https://docs.python.org/2.7/'><img src='https://img.shields.io/badge/python-2.7-blue.svg'></a>
<a href='https://docs.python.org/3.5/'><img src='https://img.shields.io/badge/python-3.5-blue.svg'></a>


About
-----


License
-------

DensPart is distributed under GPL License version 3 (GPLv3).


Dependencies
------------

The following dependencies will be necessary for DensPart to build properly,

* Python >= 2.7, >= 3.5: http://www.python.org/
* SciPy >= 0.11.0: http://www.scipy.org/
* NumPy >= 1.9.1: http://www.numpy.org/
* Nosetests >= 1.1.2: http://readthedocs.org/docs/nose/en/latest/
* HORTON >= 2.0.1: http://theochem.github.io/horton/2.0.1/index.html

The dependence on HORTON is only because of the `grid` module. This will be replaced by `qcgrids` in
the near future.


Installation
------------

To install DensPart:

```bash
python ./setup install --user
```


Testing
-------

To run tests:

```bash
nosetests -v denspart
```
