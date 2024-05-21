import numpy as np
import scipy


def complex_integration(
    f: callable,
    param: callable,
    param_jac: callable,
    limits: tuple[float, float] = (0, 1),
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

    return scipy.integrate.quad(_f, *limits, complex_func=True)[0]
