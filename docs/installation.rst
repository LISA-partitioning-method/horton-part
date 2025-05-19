.. _usr_installation:

Installation
############

Downloading the Code
====================

The latest code can be obtained from GitHub:

.. code-block:: bash

   git clone https://github.com/LISA-partitioning-method/horton-part.git

Installing
==========

To install the latest version of `HORTON-PART`.

.. code-block:: bash

   pip install horton-part

To install `HORTON-PART` version `0.0.X`:

.. code-block:: bash

   pip install horton-part==0.0.X

To install the latest version from the repository:

.. code-block:: bash

   git clone https://github.com/yingxingcheng/horton-part.git
   cd horton-part
   pip install .

Running Tests
=============

If you want to run the tests, install the test dependencies:

.. code-block:: bash

   pip install .[tests]

For developers who require all dependencies, install them with:

.. code-block:: bash

   pip install -e .[dev,tests]

Building the Documentation
==========================

To build the documentation using Sphinx into the `_build` directory:

.. code-block:: bash

    cd ./doc
    ./gen_api.sh
    sphinx-build -b html . _build

Dependencies
============

The following dependencies are required for `HORTON-PART` to build properly:

- `quadprog>=0.1.13` : https://github.com/quadprog/quadprog
- `cvxopt>=1.3.1` : https://github.com/cvxopt/cvxopt
- `qc-grid>=0.0.9` : https://github.com/theochem/grid
- `qc-iodata` : https://github.com/theochem/iodata
- `qc-gbasis>=0.1.0` : https://github.com/theochem/gbasis
