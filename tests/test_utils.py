import numpy as np
from numpy.testing import assert_equal, assert_allclose

from poles_roots._utils import (
    convert_cart_to_complex,
    parametrise_between_two_points,
    point_in_triangle,
    compute_incenter,
)


def test_convert_cart_to_complex():
    points = np.array([[1, 1], [10.7, -8], [-99.8, 7]])
    z = convert_cart_to_complex(points)
    assert_equal(z, [1 + 1j, 10.7 - 8j, -99.8 + 7j])


def test_parametrise_between_two_points():
    a = 0
    b = 1 + 1j

    param, jac = parametrise_between_two_points(a, b)

    assert_equal(param(np.array([0, 0.5, 1])), [0, 0.5 + 0.5j, 1 + 1j])
    assert_equal(jac, 1 + 1j)


def test_point_in_triangle():
    assert point_in_triangle(
        np.array([0.1, 0.1]), np.array([0, 0]), np.array([1, 0]), np.array([0, 1])
    )
    assert not point_in_triangle(
        np.array([-0.1, 0.1]), np.array([0, 0]), np.array([1, 0]), np.array([0, 1])
    )
    assert not point_in_triangle(
        np.array([1, 1]), np.array([0, 0]), np.array([1, 0]), np.array([0, 1])
    )


def test_compute_incenter():
    assert_allclose(
        compute_incenter(np.array([3, 1]), np.array([0, 3]), np.array([-3, 1])),
        np.array([0, (2 * np.sqrt(13) + 18) / (6 + 2 * np.sqrt(13))]),
        atol=1e-12,
        rtol=1e-12,
    )
