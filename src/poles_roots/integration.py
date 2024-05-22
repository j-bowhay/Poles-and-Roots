from typing import Optional

import numpy as np
import scipy


def complex_integration(
    f: callable,
    param: callable,
    param_jac: callable,
    limits: tuple[float, float] = (0, 1),
    *,
    quad_kwargs: Optional[dict] = None,
) -> complex:
    """Complex path integration

    Parameters
    ----------
    f : callable
        Integrand
    param : callable
        Parametrisation of the path
    param_jac : callable
        Derivative of the parametrisation
    limits : tuple[float, float], optional
        Limits of integration, by default (0, 1)

    Returns
    -------
    complex
        Value of the integral
    """

    def _f(t):
        return f(param(t)) * param_jac(t)

    quad_kwargs = {} if quad_kwargs is None else quad_kwargs

    return scipy.integrate.quad(_f, *limits, complex_func=True, **quad_kwargs)[0]


def argument_principal(
    f: callable,
    f_jac: callable,
    param: callable,
    param_jac: callable,
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
