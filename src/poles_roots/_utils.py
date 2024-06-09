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


def compute_incenter(A, B, C):
    """Compute the incenter of triangle ABC."""
    a, b, c = np.linalg.norm([B - C, A - C, A - B], axis=-1)

    return (a * A + b * B + c * C) / (a + b + c)


def linspace_on_tri(points, num):
    if points.shape != (3, 2):
        raise ValueError("`points` must have shape")
    points = np.vstack([points, points[0, :]])
    ds = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.cumsum(np.append(0, ds))
    b = scipy.interpolate.make_interp_spline(s, points, k=1)
    return b(np.linspace(0, s[-1], num=num, endpoint=False))


def area_2(a, b, c):
    """Compute twice the area of a triangle"""
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])


def left(point, a, b):
    """Determine if point is to the left of the line ab"""
    return area_2(a, b, point) > 0


def left_on(point, a, b):
    """Determine if point is to the left of or on the line ab"""
    return area_2(a, b, point) >= 0


def collinear(point, a, b):
    """Determine if point is on the line ab"""
    return area_2(a, b, point) == 0


def point_in_polygon(point, points, on=True):
    """Determine if point in the convex polygon defined by points"""
    for i, a in enumerate(points):
        b = np.take(points, i + 1, axis=0, mode="wrap")

        if on:
            if not left_on(point, a, b):
                return False
        else:
            if not left(point, a, b):
                return False
    return True
