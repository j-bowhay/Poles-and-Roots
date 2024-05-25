from typing import Optional, Callable

import numpy as np
import scipy


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
    """Compute the argument principal integral."""

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
    res = 0
    for i, a in enumerate(points):
        b = np.take(points, i + 1, mode="wrap")

        def param(t):
            return a * (1 - t) + b * t

        param_jac = b - a

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
    result = np.empty(simplices.shape[0], dtype=np.complex128)
    for i, simplex in enumerate(simplices):
        simplex_coords = points[simplex].T
        simplex_points = np.empty(3, dtype=np.complex128)
        simplex_points.real, simplex_points.imag = simplex_coords
        result[i] = argument_principle_from_points(
            f,
            f_jac,
            simplex_points,
            quad_kwargs,
        )
    return result
