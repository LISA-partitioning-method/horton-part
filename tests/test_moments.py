import numpy as np
import pytest
from horton_part import get_npure_cumul, fill_pure_polynomials


def test_fill_pure_polynomials():
    npoint = 10
    lmax = 4
    npure = get_npure_cumul(lmax)
    output = np.zeros((npoint, npure - 1))
    output[:, :3] = np.random.random((npoint, 3))
    z = output[:, 0]
    x = output[:, 1]
    y = output[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    expected = [
        # l = 1
        z,
        x,
        y,
        # l = 2
        (3.0 * z**2 - r**2) / 2.0,
        3.0**0.5 * x * z,
        3.0**0.5 * y * z,
        3.0**0.5 * (x * x - y * y) / 2.0,
        3.0**0.5 * x * y,
        # l = 3
        (5.0 * z**3 - 3.0 * r**2.0 * z) / 2,
        x * (15.0 * z**2 - 3.0 * r**2) / 2.0 / 6.0**0.5,
        y * (15.0 * z**2 - 3.0 * r**2) / 2.0 / 6.0**0.5,
        z * (x**2 - y**2) * 15.0**0.5 / 2,
        x * y * z * 15.0**0.5,
        10.0**0.5 / 4 * x * (x**2 - 3.0 * y**2),
        10.0**0.5 / 4 * y * (3.0 * x**2 - y**2),
        # l = 4
        (3.0 * r**4 - 30.0 * r**2 * z**2 + 35.0 * z**4) / 8,
        x * (35 * z**3 - 15.0 * r**2 * z) / 2.0 / 10.0**0.5,
        y * (35 * z**3 - 15.0 * r**2 * z) / 2.0 / 10.0**0.5,
        (105.0 * z**2 - 15.0 * r**2) * (x**2 - y**2) / 2.0 / 30.0 * 5.0**0.5,
        x * y * (105.0 * z**2 - 15.0 * r**2) / 2.0 / 15.0 * 5**0.5,
        z * x * (x**2 - 3 * y**2) * 70**0.5 / 4.0,
        z * y * (3 * x**2 - y**2) * 70**0.5 / 4.0,
        (x**4 - 6.0 * x**2 * y**2 + y**4) * 35**0.5 / 8,
        x * y * (x**2 - y**2) * 35**0.5 / 2,
    ]

    fill_pure_polynomials(output, 4)

    for i in range(npure - 1):
        assert output[:, i] == pytest.approx(expected[i])
