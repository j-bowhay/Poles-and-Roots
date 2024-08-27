from dataclasses import dataclass
from typing import Callable

import numpy as np
from tqdm import tqdm

from poles_roots.triangulation import adaptive_triangulation
from poles_roots.aaa import AAA
from poles_roots._utils import (
    convert_cart_to_complex,
    point_in_triangle,
    points_in_triangle,
)


@dataclass
class ZerosPolesResult:
    roots: np.ndarray
    poles: np.ndarray
    residues: np.ndarray
    points: np.ndarray
    simplices: np.ndarray


def find_zeros_poles(
    f: Callable,
    f_prime: Callable,
    initial_points: np.ndarray,
    num_sample_points: int,
    arg_principal_threshold: float,
    quad_kwargs=None,
    cross_ref=True,
    rng=None,
) -> ZerosPolesResult:
    """Compute all the zeros and pole of `f`

    Parameters
    ----------
    f : Callable
        Function to compute poles and zeros of.
    f_prime : Callable
        Derivative of `f`.
    initial_points : array
        Points describing the search region.
    num_sample_points : int
        Number of points to sample on the boundary of each simplex for aaa.
    arg_principal_threshold : float
        Threshold value for Cauchy's argument principle for the adaptive triangulation
    quad_kwargs : dict, optional
        arguments to be passed to `scipy.integrate.quad`, by default None

    Returns
    -------
    _ZerosPolesResult
        Results object
    """
    points = initial_points

    # iterate until AAA and arg principle agree
    while True:
        # triangulate the domain
        tri, arg_princ_z_minus_ps = adaptive_triangulation(
            f,
            f_prime,
            points,
            arg_principal_threshold,
            quad_kwargs=quad_kwargs,
            rng=rng,
        )

        poles = []
        residues = []
        zeros = []
        refine_further = False
        new_points = tri.points

        # apply aaa on each simplex
        for simplex, arg_princ_z_minus_p in tqdm(
            zip(tri.simplices, arg_princ_z_minus_ps)
        ):
            # generate points on the edge of the simplex
            sample_points = points_in_triangle(
                *tri.points[simplex, :], num_sample_points
            ).T
            z = convert_cart_to_complex(sample_points)

            # sample the function on the edge of the simplex
            F = f(z)

            # only report zeros and poles that are within the simplex
            simplex_points = tri.points[simplex, :]

            aaa_z_minus_p = 0

            aaa_log_deriv = AAA(f_prime(z) / F, z)

            for pole, residue in zip(aaa_log_deriv.poles, aaa_log_deriv.residues):
                if (
                    point_in_triangle(np.array([pole.real, pole.imag]), *simplex_points)
                    and not np.isclose(residue, 0)
                    and np.isclose(round(residue.real), residue)
                ):
                    aaa_z_minus_p += round(residue.real)
                    if residue > 0:
                        zeros.append(pole)
                    else:
                        poles.append(pole)

            # report if AAA and argument principle are not matching and destroy the
            # triangle if so
            if cross_ref and not np.allclose(arg_princ_z_minus_p, aaa_z_minus_p):
                print(
                    f"AAA and argument principle don't match: {aaa_z_minus_p=}, {arg_princ_z_minus_p=}"
                )
                refine_further = True
                new_points = np.concatenate(
                    [new_points, simplex_points.mean(axis=0)[np.newaxis]]
                )

        if not refine_further:
            return ZerosPolesResult(
                roots=np.asarray(zeros),
                poles=np.asarray(poles),
                residues=np.asarray(residues),
                points=tri.points,
                simplices=tri.simplices,
            )
        else:
            points = convert_cart_to_complex(new_points)
