from typing import Optional, Callable

import numpy as np
import scipy

from poles_roots._utils import convert_cart_to_complex, parametrise_between_two_points


def complex_integration(
    f: Callable,
    param: Callable,
    param_jac: Callable | float,
    limits: tuple[float, float] = (0, 1),
    *,
    quad_kwargs: Optional[dict] = None,
) -> complex:
    """Complex path integration"""
    if callable(param_jac):

        def _f(t):
            return f(param(t)) * param_jac(t)
    else:

        def _f(t):
            return f(param(t)) * param_jac

    quad_kwargs = {} if quad_kwargs is None else quad_kwargs

    return scipy.integrate.quad(_f, *limits, complex_func=True, **quad_kwargs)[0]


def argument_principle_from_parametrisation(
    f: Callable,
    f_jac: Callable,
    param: Callable,
    param_jac: Callable | float,
    limits: tuple[float, float],
    quad_kwargs: Optional[dict] = None,
) -> float:
    """Compute the argument principal integral of a parametrised curve."""

    def _f(t):
        return f_jac(t) / f(t)

    return complex_integration(
        _f,
        param,
        param_jac,
        limits,
        quad_kwargs=quad_kwargs,
    ) / (2 * np.pi * 1j)


def argument_principle_from_points(
    f: Callable,
    f_jac: Callable,
    points: np.ndarray,
    quad_kwargs: Optional[dict] = None,
) -> complex:
    """Compute the argument principle of a closed curve defined by line segments between
    the given points."""
    res = 0
    for i, a in enumerate(points):
        b = np.take(points, i + 1, mode="wrap")

        param, param_jac = parametrise_between_two_points(a, b)

        res += argument_principle_from_parametrisation(
            f,
            f_jac,
            param,
            param_jac,
            (0, 1),
            quad_kwargs=quad_kwargs,
        )

    return res


def argument_priciple_of_triangulation(
    f,
    f_jac,
    points,
    simplices,
    quad_kwargs: Optional[dict] = None,
):
    """Computes the argument principle around each simplex in a triangulation."""
    result = np.empty(simplices.shape[0], dtype=np.complex128)
    for i, simplex in enumerate(simplices):
        simplex_points = convert_cart_to_complex(points[simplex])
        result[i] = argument_principle_from_points(
            f,
            f_jac,
            simplex_points,
            quad_kwargs,
        )
    return result
