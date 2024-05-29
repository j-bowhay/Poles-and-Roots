from dataclasses import dataclass
from typing import Callable

import numpy as np

from poles_roots.triangulation import adaptive_triangulation
from poles_roots.aaa import AAA
from poles_roots.utils import (
    convert_cart_to_complex,
    point_in_triangle,
    linspace_on_tri,
)


@dataclass
class _ZerosPolesResult:
    zeros: np.ndarray
    poles: np.ndarray
    residuals: np.ndarray
    points: np.ndarray
    simplices: np.ndarray


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
        Number of points to sample on the boundary of each simplex for aaa.

        TODO: should probably do something more sophisticated accounting for the size
        of the simplex

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
    points, simplices, arg_princ_z_minus_ps = adaptive_triangulation(
        f, f_jac, points, arg_principal_threshold, quad_kwargs
    )

    poles = []
    residuals = []
    zeros = []
    # apply aaa on each simplex
    for simplex, arg_princ_z_minus_p in zip(simplices, arg_princ_z_minus_ps):
        # generate points on the edge of the simplex
        sample_points = linspace_on_tri(points[simplex, :], num_sample_points)
        z = convert_cart_to_complex(sample_points)

        # function values
        F = f(z)

        aaa_res = AAA(F, z)

        # only report zeros and poles that are within the simplex
        A, B, C = points[simplex, :]

        aaa_z_minus_p = 0
        for pole, residual in zip(aaa_res.poles, aaa_res.residuals):
            if point_in_triangle(np.array([pole.real, pole.imag]), A, B, C):
                poles.append(pole)
                residuals.append(residual)
                aaa_z_minus_p -= 1

        for zero in aaa_res.zeros:
            if point_in_triangle(np.array([zero.real, zero.imag]), A, B, C):
                zeros.append(zero)
                aaa_z_minus_p += 1

        if not np.allclose(arg_princ_z_minus_p, aaa_z_minus_p):
            print(
                f"AAA and argument principle don't match: {aaa_z_minus_p=}, {arg_princ_z_minus_p=}"
            )
            # TODO: do further refinement

    return _ZerosPolesResult(
        zeros=np.asarray(zeros),
        poles=np.asarray(poles),
        residuals=np.asarray(residuals),
        points=points,
        simplices=simplices,
    )
