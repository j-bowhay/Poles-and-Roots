import numpy as np


def convert_cart_to_complex(points):
    """Converts 2d array of coordinates to a 1d array of complex numbers"""
    z = np.empty(points.shape[0], dtype=np.complex128)
    z.real, z.imag = points.T
    return z


def parametrise_between_two_points(a, b):
    """Returns a parametrisation between points `a` and `b` and its derivative."""

    def param(t):
        return a * (1 - t) + b * t

    return param, b - a
