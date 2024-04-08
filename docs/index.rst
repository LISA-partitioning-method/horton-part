.. image:: horton_part_logo.svg
  :width: 450
  :align: center


Welcome to Horton-Part's documentation!
=======================================

`HORTON-PART <https://github.com/yingxingcheng/horton-part>`_ is a computational chemistry package that supports different partition schemes. It is based on the sub-module ``part`` of ``HORTON2``, which is written and maintained by Toon Verstraelen (1).
In HORTON3, all sub-modules have been rewritten using the pure Python programming language to support Python 3+. The 'part' module has also been rewritten and is now called `denspart <https://github.com/theochem/denspart>`_ module.
However, the algorithm implemented in ``denspart`` only uses one-step optimization, which can be computationally expensive for large systems. Additionally, ``denspart`` only supports the ``MBIS`` partitioning scheme.
Another ``part`` module has been rewritten in pure Python programming language by Farnaz Heidar-Zadeh (2). However, the integration grid implemented in this module still uses the old 'grid' from Horton2. `HORTON-PART <https://github.com/yingxingcheng/horton-part>`_ with version ``0.0.X`` is based on this module. Starting from version ``1.X.X``, `HORTON-PART <https://github.com/yingxingcheng/horton-part>`_ supports the new integration `qc-grid <https://github.com/theochem/grid>`_.

This version contains contributions from YingXing Cheng (4), Toon Verstraelen (1), Pawel Tecmer (2), Farnaz Heidar-Zadeh (2), Cristina E. González-Espinoza (2), Matthew Chan (2), Taewon D. Kim (2), Katharina Boguslawski (2), Stijn Fias (3), Steven Vandenbrande (1), Diego Berrocal (2), and Paul W. Ayers (2)

- (1) Center for Molecular Modeling (CMM), Ghent University, Ghent, Belgium.
- (2) The Ayers Group, McMaster University, Hamilton, Ontario, Canada.
- (3) General Chemistry (ALGC), Free University of Brussels, Brussels, Belgium.
- (4) Numerical Mathematics for High Performance Computing (NMH), University of Stuttgart, Stuttgart, Germany.

More information about HORTON can be found on this `website <http://theochem.github.com/horton/>`_.

..
    Please use the following citation in any publication using horton-part library:

        **"Grid: A Python Library for Molecular Integration, Interpolation and More."**,
        X. D. Yang,  L. Pujal, A. Tehrani,  R. Hernandez‐Esparza,  E. Vohringer‐Martinez,
        T. Verstraelen, P. W. Ayers, F. Heidar‐Zadeh


The Horton-Part source code is hosted on GitHub and is released under the GNU General Public License v3.0. We welcome
any contributions to the Horton-Part library in accordance with our Code of Conduct; please see our Contributing
Guidelines. Please report any issues you encounter while using Horton-Part library on
`GitHub Issues <https://github.com/yingxingcheng/horton-part/issues/new>`_. For further
information and inquiries please contact us at yxcheng2buaa@gmail.com.


Functionality
=============
* Real space partitioning schemes

  * Mulliken partitioning method

* Density related partitioning schemes

  * Becke partitioning method

  * Stockholder schemes

    * Hirshfeld partitioning method

  * Iterative Stockholder schemes

    * Iterative Hirshfeld (Hirshfeld-I) partitioning method

    * Iterative Stockholder Approach (ISA)

    * Minimal Basis Iterative Stockholder (MBIS)

    * Gaussian iterative stockholder approach (GISA)

    * Local Linear approximation of the ISA (LISA) method

    * Global Linear approximation of the ISA (GLISA) method



Modules
=======

.. csv-table::
    :file: ./table_modules.csv
    :widths: 50, 70
    :delim: ;
    :header-rows: 1
    :align: center


.. toctree::
    :maxdepth: 2
    :caption: User Documentation

    ./installation.rst



.. toctree::
    :maxdepth: 1
    :caption: Quick Start

    ./notebooks/quick_start.ipynb


.. toctree::
    :maxdepth: 1
    :caption: Example Tutorials

    ./notebooks/setup.rst
    ./notebooks/isa.ipynb
    ./notebooks/gisa.ipynb
    ./notebooks/mbis.ipynb
    ./notebooks/lisa.ipynb
    ./notebooks/lisa_diis.ipynb
    ./notebooks/lisa_g.ipynb

    ./notebooks/hirshfeld.ipynb
    ./notebooks/mulliken.ipynb


.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   pyapi/modules.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
