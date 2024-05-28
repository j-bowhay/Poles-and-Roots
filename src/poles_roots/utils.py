import numpy as np
import scipy


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


def point_in_triangle(point, A, B, C):
    """Check if a point lies in the triangle ABC"""
    # TODO: vectorise
    p = point - A
    c = C - A
    b = B - A

    lhs = np.array([[np.dot(c, c), np.dot(b, c)], [np.dot(b, c), np.dot(b, b)]])

    rhs = np.array([np.dot(p, c), np.dot(p, b)])

    u, v = scipy.linalg.solve(lhs, rhs)

    return u > 0 and v > 0 and 1 - u - v > 0
