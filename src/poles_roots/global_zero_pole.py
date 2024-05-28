from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from poles_roots.triangulation import adaptive_triangulation
from poles_roots.aaa import AAA
from poles_roots.utils import (
    convert_cart_to_complex,
    parametrise_between_two_points,
    point_in_triangle,
)


@dataclass
class _ZerosPolesResult:
    points: np.ndarray
    simplices: np.ndarray
    zeros: np.ndarray = field(default_factory=lambda: np.array([]))
    poles: np.ndarray = field(default_factory=lambda: np.array([]))


def find_zeros_poles(
    f: Callable,
    f_jac: Callable,
    points: np.ndarray,
    num_sample_points: int,
    arg_principal_threshold: float,
    quad_kwargs=None,
) -> _ZerosPolesResult:
    """Compute all the zeros and pole of `f`

    Parameters
    ----------
    f : Callable
        Function to compute poles and zeros of.
    f_jac : Callable
        Derivative of `f`.
    points : array
        Points describing the Jordan curve to search the interior of.
    num_sample_points : int
        Number of points to sample on each side of the simplex for aaa.

        TODO: should probably do something more sophisticated accounting for side length etc

    arg_principal_threshold : float
        Threshold value for Cauchy's argument principle for the adaptive triangulation
    quad_kwargs : dict, optional
        arguments to be passed to `scipy.integrate.quad`, by default None

    Returns
    -------
    _ZerosPolesResult
        _description_
    """
    # triangulate the domain
    points, simplices = adaptive_triangulation(
        f, f_jac, points, arg_principal_threshold, quad_kwargs
    )

    res = _ZerosPolesResult(points=points, simplices=simplices)
    # apply aaa on each simplex
    for simplex in simplices:
        simplex_points = convert_cart_to_complex(points[simplex, :])

        # generate points on the edge of the simplex
        z = np.empty(3 * num_sample_points, dtype=np.complex128)
        for i, a in enumerate(simplex_points):
            b = np.take(simplex_points, i + 1, mode="wrap")

            param, _ = parametrise_between_two_points(a, b)
            z[i * num_sample_points : (i + 1) * num_sample_points] = param(
                np.linspace(0, 1, num=num_sample_points, endpoint=False)
            )

        # function values
        F = f(z)

        aaa_res = AAA(F, z)

        # only report zeros and poles that are within the simplex
        A, B, C = points[simplex, :]

        for pole in aaa_res.poles:
            if point_in_triangle(np.array([pole.real, pole.imag]), A, B, C):
                res.poles = np.append(res.poles, pole)

        for zero in aaa_res.zeros:
            if point_in_triangle(np.array([zero.real, zero.imag]), A, B, C):
                res.zeros = np.append(res.zeros, zero)

    return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from poles_roots.reference_problems import func2, func2_jac
    from poles_roots.plotting import phase_plot, plot_poles_zeros

    res = find_zeros_poles(
        func2,
        func2_jac,
        points=[-9 - 9.2j, 9 - 10j, 9.9 + 9.8j, -9.3 + 9.5j],
        arg_principal_threshold=1.1,
        num_sample_points=50,
    )

    fig, ax = plt.subplots()
    phase_plot(func2, ax, domain=[-10, 10, -10, 10])
    plot_poles_zeros(res, ax)
    ax.legend()
    plt.show()
