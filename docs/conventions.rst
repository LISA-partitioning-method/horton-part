.. _conventions:

Conventions
############

Spherical Coordinates
=====================

Spherical coordinates :math:`(r, \theta, \phi)` is used extensively throughout the molecular, atomic,
and angular grid modules.
Particularly, the radius :math:`r \in [0, \infty)`, azumuthal :math:`\theta \in [-\pi, \pi]` and polar
:math:`\phi \in [0, \pi)` angles are defined from Cartesian coordinates as:

.. math::
    \begin{align}
        r &= \sqrt{x^2 + y^2 + z^2}\\
        \theta &= \text{arctan2} \bigg(\frac{y}{x}\bigg)\\
        \phi &= arc\cos \bigg(\frac{z}{r}\bigg)
    \end{align}

such that when the radius is zero :math:`r=0`, then the angles are zero :math:`\theta,\phi = 0`.

Grid offers a :class:`utility<grid.utils.convert_cart_to_sph>` function to convert to spherical coordinates

.. code-block::
    python

    import numpy as np
    from grid.utils import convert_cart_to_sph

    # Generate random set of points
    cart_pts = np.random.uniform((-100, 100), size=(1000, 3))
    # Convert to spherical coordinates
    spher_pts = convert_cart_to_sph(cart_pts)
    # Convert to spherical coordinates, centered at [1, 1, 1]
    spher_pts = convert_cart_to_sph(cart_pts, center=np.array([1.0, 1.0, 1.0]))


The :class:`atomic grid<grid.atomgrid.AtomGrid>` class can also be used to convert its points to spherical coordinates

.. code-block::
    python

    from grid.atomgrid import AtomGrid

    # Construct atomic grid object
    atom_grid = AtomGrid(...)

    # Convert atomic grid points centered at the nucleus to spherical coordinates
    spher_pts = atom_grid.convert_cartesian_to_spherical()

With the convention that when :math:`r=0`, then the angles within :math:`\{(0, \theta_j, \phi_j)\}` in that shell is
obtained from a angular grid :math:`\{(\theta_j, \phi_j)\}` with some degree.

Spherical Harmonics
===================

The real spherical harmonics are defined using complex spherical harmonics:

.. math::
    Y_{lm}(\theta, \phi) = \begin{cases}
        i(Y^m_l(\theta, \phi) - (-1)^m Y_l^{-m}(\theta, \phi) & \text{if } m < 0 \\
        Y_l^0 & \text{if } m = 0 \\
        (Y^{-|m|}_{l}(\theta, \phi) + (-1)^m Y_l^m(\theta, \phi)) & \text{if } m > 0
    \end{cases},

where the degree :math:`l \in \mathbb{N}`, order :math:`m \in \{-l, \cdots, l \}` and
:math:`Y^m_l` is the complex spherical harmonic.  The Condon-Shortley phase is not included.


Alternatively, it can be written using the associated Legendre polynomials :math:`P_l^m` (without the Conway phase):

.. math::
    Y_{lm}(\theta, \phi) = \begin{cases}
        \sqrt{\frac{(2l + 1) (l - m)!}{4 \pi (l + m)!}} \sqrt{2} \cos(m \theta) P_l^m(\cos(\theta)) & \text{if } m < 0 \\
        \frac{1}{2} \sqrt{\frac{1}{\pi}} & \text{if } m = 0 \\
        \sqrt{\frac{(2l + 1) (l - m)!}{4 \pi (l + m)!}}  \sqrt{2}\sin(m \theta) P_l^m(\cos(\theta))  & \text{if } m > 0
    \end{cases}

Grid offers :func:`generate_real_spherical_harmonics<grid.utils.generate_real_spherical_harmonics>` function
to generate the real spherical harmonics:

.. code-block::
    python

    from grid.utils import generate_real_spherical_harmonics

    spher_pts = np.array(...)
    theta = spher_pts[:, 1]
    phi = spher_pts[:, 2]
    # Generate all degrees up to l=2
    spherical_harmonics = generate_real_spherical_harmonics(2, theta, phi)


Ordering
--------

The spherical harmonics are first ordered by the degree :math:`l` in ascending order.

For each degree :math:`l`, the orders :math:`m` are in HORTON2 order defined as:

.. math::
   m = [0, 1, -1, 2, -2, \cdots, l, -l].
