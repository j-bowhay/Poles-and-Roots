from typing import Optional, Callable

import numpy as np
import scipy
import scipy.integrate

from poles_roots._utils import convert_cart_to_complex, parametrise_between_two_points


def complex_integration(
    f: Callable,
    param: Callable,
    param_prime: Callable | float,
    limits: tuple[float, float] = (0, 1),
    *,
    method: str = "quad",
    quad_kwargs: Optional[dict] = None,
) -> complex:
    """Complex path integration"""
    # parametrise the integral
    if callable(param_prime):

        def _f(t):
            return f(param(t)) * param_prime(t)
    else:

        def _f(t):
            return f(param(t)) * param_prime

    quad_kwargs = {} if quad_kwargs is None else quad_kwargs

    if method == "quad":
        return scipy.integrate.quad(
            _f, *limits, complex_func=True, full_output=True, **quad_kwargs
        )[0]
    elif method == "fixed":
        return scipy.integrate.fixed_quad(_f, *limits, **quad_kwargs)[0]
    elif method == "trapezium":
        t = np.linspace(*limits, **quad_kwargs)
        return scipy.integrate.trapezoid(_f(t), t)
    else:
        raise ValueError("Invalid Method")


def argument_principle_from_parametrisation(
    f: Callable,
    f_prime: Callable,
    param: Callable,
    param_prime: Callable | float,
    limits: tuple[float, float],
    quad_kwargs: Optional[dict] = None,
    method="quad",
    moment=0,
) -> float:
    """Compute the argument principal integral of a parametrised curve."""

    def _f(t):
        with np.errstate(divide="ignore", invalid="ignore"):
            return t**moment * f_prime(t) / f(t)

    return complex_integration(
        _f, param, param_prime, limits, quad_kwargs=quad_kwargs, method=method
    ) / (2 * np.pi * 1j)


def argument_principle_from_points(
    f: Callable,
    f_prime: Callable,
    points: np.ndarray,
    quad_kwargs: Optional[dict] = None,
    method="quad",
    moment=0,
) -> complex:
    """Compute the argument principle of a closed curve defined by line segments between
    the given points."""
    inf_edges = set()
    res = 0

    # iterate over the edges
    for i, a in enumerate(points):
        b = np.take(points, i + 1, mode="wrap")

        param, param_prime = parametrise_between_two_points(a, b)

        edge_res = argument_principle_from_parametrisation(
            f,
            f_prime,
            param,
            param_prime,
            (0, 1),
            quad_kwargs=quad_kwargs,
            method=method,
            moment=moment,
        )
        res += edge_res

        # if the integration hasn't worked we'll need to destroy this edge
        if np.isnan(edge_res) or np.isinf(edge_res):
            # we a frozen set here because were integrate each edge twice but in
            # opposite directions we only want to destroy it once
            inf_edges.add(frozenset((a, b)))

    return res, inf_edges


def argument_priciple_of_triangulation(
    f,
    f_prime,
    points,
    simplices,
    quad_kwargs: Optional[dict] = None,
):
    """Computes the argument principle around each simplex in a triangulation."""
    result = np.empty(simplices.shape[0], dtype=np.complex128)
    inf_edges_global = set()
    for i, simplex in enumerate(simplices):
        simplex_points = convert_cart_to_complex(points[simplex])
        res, inf_edges = argument_principle_from_points(
            f,
            f_prime,
            simplex_points,
            quad_kwargs,
        )
        result[i] = res
        # keep track of the edges that we are going to destroy
        inf_edges_global.update(inf_edges)
    return result, inf_edges_global
