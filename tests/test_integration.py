import pytest
import numpy as np
import scipy
from numpy.testing import assert_allclose

from poles_roots.integration import (
    complex_integration,
    argument_principle_from_parametrisation,
    argument_principle_from_points,
    argument_priciple_of_triangulation,
)


def quadratic(z):
    return z**2


def param_1(t):
    return t * (1 + 1j)


def param_1_jac(t):
    return 1 + 1j


def trig(z):
    return np.cos(z)


def param_2(t):
    return t * 1j


def param_2_jac(t):
    return 1j


def z_inv(z):
    return 1 / z


def z_inv_jac(z):
    return -1 / z**2


def param_3(t):
    return np.exp(t * 1j)


def param_3_jac(t):
    return 1j * np.exp(t * 1j)


@pytest.mark.parametrize(
    "f,param,param_jac,limits,expected",
    [
        (quadratic, param_1, param_1_jac, (0, 1), (2 / 3) * (-1 + 1j)),
        (quadratic, param_1, 1 + 1j, (0, 1), (2 / 3) * (-1 + 1j)),
        (trig, param_2, param_2_jac, (-np.pi, np.pi), 23.097479j),
        (trig, param_2, 1j, (-np.pi, np.pi), 23.097479j),
        (z_inv, param_3, param_3_jac, (0, 2 * np.pi), 2 * np.pi * 1j),
    ],
)
def test_complex_integration(f, param, param_jac, limits, expected):
    assert_allclose(complex_integration(f, param, param_jac, limits), expected)


@pytest.mark.parametrize(
    "f,f_jac,param,param_jac,limits,expected",
    [
        (z_inv, z_inv_jac, param_3, param_3_jac, (0, 2 * np.pi), -1),
        (lambda z: z, lambda z: 1, param_3, param_3_jac, (0, 2 * np.pi), 1),
    ],
)
def test_argument_principal(f, f_jac, param, param_jac, limits, expected):
    assert_allclose(
        argument_principle_from_parametrisation(
            f,
            f_jac,
            param,
            param_jac,
            limits=limits,
        ),
        expected,
    )


@pytest.mark.parametrize(
    "f,f_jac,points,expected",
    [
        (z_inv, z_inv_jac, [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j], -1),
        (lambda z: z, lambda z: 1, [-10 - 10j, 1 - 1j, 20 + 1j, -1 + 1j], 1),
    ],
)
def test_argument_principal_from_points(f, f_jac, points, expected):
    assert_allclose(
        argument_principle_from_points(f, f_jac, points),
        expected,
    )


def test_argument_principle_from_triangulation():
    def f(z):
        return (z - 1) / (z + 1)

    def f_prime(z):
        return 2 / (z + 1) ** 2

    points = np.array([[0, 1], [0, -1], [5, 0], [-5, 0]])

    tri = scipy.spatial.Delaunay(points)

    z_minus_p = argument_priciple_of_triangulation(
        f, f_prime, tri.points, tri.simplices
    )

    assert_allclose(z_minus_p, [-1 + 0j, 1 + 0j])
