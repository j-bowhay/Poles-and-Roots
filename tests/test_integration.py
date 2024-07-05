import pytest
import numpy as np
import scipy
from numpy.testing import assert_allclose, assert_equal

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


def param_1_prime(t):
    return 1 + 1j


def trig(z):
    return np.cos(z)


def param_2(t):
    return t * 1j


def param_2_prime(t):
    return 1j


def z_inv(z):
    return 1 / z


def z_inv_prime(z):
    return -1 / z**2


def param_3(t):
    return np.exp(t * 1j)


def param_3_prime(t):
    return 1j * np.exp(t * 1j)


@pytest.mark.parametrize(
    "f,param,param_prime,limits,expected",
    [
        (quadratic, param_1, param_1_prime, (0, 1), (2 / 3) * (-1 + 1j)),
        (quadratic, param_1, 1 + 1j, (0, 1), (2 / 3) * (-1 + 1j)),
        (trig, param_2, param_2_prime, (-np.pi, np.pi), 23.097479j),
        (trig, param_2, 1j, (-np.pi, np.pi), 23.097479j),
        (z_inv, param_3, param_3_prime, (0, 2 * np.pi), 2 * np.pi * 1j),
    ],
)
@pytest.mark.parametrize("method", ["quad", "fixed", "trapezium"])
def test_complex_integration(f, param, param_prime, limits, expected, method):
    if method == "fixed":
        quad_kwargs = {"n": 20}
    elif method == "trapezium":
        quad_kwargs = {"num": 10000}
    else:
        quad_kwargs = None
    assert_allclose(
        complex_integration(
            f, param, param_prime, limits, method=method, quad_kwargs=quad_kwargs
        ),
        expected,
    )


@pytest.mark.parametrize(
    "f,f_prime,param,param_prime,limits,expected",
    [
        (z_inv, z_inv_prime, param_3, param_3_prime, (0, 2 * np.pi), -1),
        (lambda z: z, lambda z: 1, param_3, param_3_prime, (0, 2 * np.pi), 1),
    ],
)
@pytest.mark.parametrize("method", ["quad", "fixed", "trapezium"])
def test_argument_principal(f, f_prime, param, param_prime, limits, expected, method):
    assert_allclose(
        argument_principle_from_parametrisation(
            f, f_prime, param, param_prime, limits=limits, method=method
        ),
        expected,
    )


class TestArgumentPrincipleFromPoints:
    @pytest.mark.parametrize(
        "f,f_prime,points,expected",
        [
            (z_inv, z_inv_prime, [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j], -1),
            (lambda z: z, lambda z: 1, [-10 - 10j, 1 - 1j, 20 + 1j, -1 + 1j], 1),
        ],
    )
    @pytest.mark.parametrize("method", ["quad", "fixed", "trapezium"])
    def test_finite(self, f, f_prime, points, expected, method):
        if method == "fixed":
            quad_kwargs = {"n": 50}
        elif method == "trapezium":
            quad_kwargs = {"num": 10000}
        else:
            quad_kwargs = None
        res, inf_edges = argument_principle_from_points(
            f, f_prime, points, method=method, quad_kwargs=quad_kwargs
        )
        assert_allclose(res, expected)
        assert_equal(inf_edges, set())

    @pytest.mark.parametrize(
        "f,f_prime,points",
        [
            (z_inv, z_inv_prime, [-1, 1, 1j]),
        ],
    )
    def test_poles(self, f, f_prime, points):
        _, inf_edges = argument_principle_from_points(f, f_prime, points)
        assert_equal(inf_edges, {frozenset((-1, 1))})

    @pytest.mark.parametrize(
        "f,f_prime,points,expected",
        [
            (lambda z: z, lambda z: 1, [-10 - 10j, 1 - 1j, 20 + 1j, -1 + 1j], 0),
            (z_inv, z_inv_prime, [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j], 0),
            (lambda z: z - 0.5, lambda z: 1, [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j], 0.5),
            (
                lambda z: 1 / (z + 0.1),
                lambda z: -1 / (z + 0.1) ** 2,
                [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j],
                0.1,
            ),
        ],
    )
    def test_moment(self, f, f_prime, points, expected):
        res, inf_edges = argument_principle_from_points(f, f_prime, points, moment=1)
        assert_allclose(res, expected, atol=1e-12)
        assert_equal(inf_edges, set())


class TestArgumentPrincipleFromTriangulation:
    def test_no_poles_on_diagonal(self):
        def f(z):
            return (z - 1) / (z + 1)

        def f_prime(z):
            return 2 / (z + 1) ** 2

        points = np.array([[0, 1], [0, -1], [5, 0], [-5, 0]])

        tri = scipy.spatial.Delaunay(points)

        z_minus_p, inf_diags = argument_priciple_of_triangulation(
            f, f_prime, tri.points, tri.simplices
        )

        assert_allclose(z_minus_p, [-1 + 0j, 1 + 0j])
        assert_equal(inf_diags, set())

    def test_pole_on_diagonal(self):
        points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])

        tri = scipy.spatial.Delaunay(points)

        z_minus_p, inf_diags = argument_priciple_of_triangulation(
            z_inv, z_inv_prime, tri.points, tri.simplices
        )

        assert_equal(inf_diags, {frozenset({(-1 + 1j), (1 - 1j)})})
