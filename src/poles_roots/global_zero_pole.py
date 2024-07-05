from dataclasses import dataclass
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from poles_roots.triangulation import adaptive_triangulation
from poles_roots.aaa import AAA
from poles_roots._utils import (
    convert_cart_to_complex,
    point_in_triangle,
    linspace_on_tri,
    compute_incenter,
)
from poles_roots.plotting import phase_plot, plot_poles_zeros


@dataclass
class ZerosPolesResult:
    zeros: np.ndarray
    poles: np.ndarray
    residues: np.ndarray
    points: np.ndarray
    simplices: np.ndarray


def find_zeros_poles(
    f: Callable,
    f_jac: Callable,
    initial_points: np.ndarray,
    num_sample_points: int,
    arg_principal_threshold: float,
    quad_kwargs=None,
    plot_triangulation=False,
    plot_aaa=False,
    approx_func="f'/f",
) -> ZerosPolesResult:
    """Compute all the zeros and pole of `f`

    Parameters
    ----------
    f : Callable
        Function to compute poles and zeros of.
    f_jac : Callable
        Derivative of `f`.
    initial_points : array
        Points describing the Jordan curve to search the interior of.
    num_sample_points : int
        Number of points to sample on the boundary of each simplex for aaa.

        TODO: should probably do something more sophisticated accounting for the size
        of the simplex

    arg_principal_threshold : float
        Threshold value for Cauchy's argument principle for the adaptive triangulation
    quad_kwargs : dict, optional
        arguments to be passed to `scipy.integrate.quad`, by default None
    plot_triangulation : bool, optional
        Plots each step of the adaptive triangulation for debugging
    plot_aaa : bool, optional
        Plots each AAA approximation for debugging

    Returns
    -------
    _ZerosPolesResult
        _description_
    """
    points = initial_points

    # iterate until AAA and arg principle agree
    # TODO: don't iterate forever!
    while True:
        # triangulate the domain
        tri, arg_princ_z_minus_ps = adaptive_triangulation(
            f,
            f_jac,
            points,
            arg_principal_threshold,
            quad_kwargs=quad_kwargs,
            plot=plot_triangulation,
        )

        poles = []
        residues = []
        zeros = []
        refine_further = False
        new_points = tri.points

        # apply aaa on each simplex
        for simplex, arg_princ_z_minus_p in zip(tri.simplices, arg_princ_z_minus_ps):
            # generate points on the edge of the simplex
            sample_points = linspace_on_tri(tri.points[simplex, :], num_sample_points)
            z = convert_cart_to_complex(sample_points)

            # sample the function on the edge of the simplex
            F = f(z)

            # only report zeros and poles that are within the simplex
            simplex_points = tri.points[simplex, :]

            aaa_z_minus_p = 0

            if approx_func == "f'/f":
                aaa_log_deriv = AAA(f_jac(z) / F, z)

                for pole, residue in zip(aaa_log_deriv.poles, aaa_log_deriv.residues):
                    if point_in_triangle(
                        np.array([pole.real, pole.imag]), *simplex_points
                    ):
                        if residue > 0:
                            zeros.append(pole)
                            aaa_z_minus_p += 1
                        else:
                            poles.append(pole)
                            aaa_z_minus_p -= 1
            elif approx_func == "f":
                aaa_f = AAA(F, z)

                for pole, residue in zip(aaa_f.poles, aaa_f.residues):
                    if point_in_triangle(
                        np.array([pole.real, pole.imag]), *simplex_points
                    ):
                        poles.append(pole)
                        residues.append(residue)
                        aaa_z_minus_p -= 1

                for zero in aaa_f.zeros:
                    if point_in_triangle(
                        np.array([zero.real, zero.imag]), *simplex_points
                    ):
                        zeros.append(zero)
                        aaa_z_minus_p += 1

            elif approx_func == "1/f":
                aaa_reciprocal = AAA(1 / F, z)

                for pole in aaa_reciprocal.zeros:
                    if point_in_triangle(
                        np.array([pole.real, pole.imag]), *simplex_points
                    ):
                        poles.append(pole)
                        aaa_z_minus_p -= 1

                for zero in aaa_reciprocal.poles:
                    if point_in_triangle(
                        np.array([zero.real, zero.imag]), *simplex_points
                    ):
                        zeros.append(zero)
                        aaa_z_minus_p += 1

            elif approx_func == "both":
                aaa_f = AAA(F, z)
                aaa_reciprocal = AAA(1 / F, z)

                for pole, residue in zip(aaa_f.poles, aaa_f.residues):
                    if point_in_triangle(
                        np.array([pole.real, pole.imag]), *simplex_points
                    ):
                        poles.append(pole)
                        residues.append(residue)
                        aaa_z_minus_p -= 1

                for zero in aaa_reciprocal.poles:
                    if point_in_triangle(
                        np.array([zero.real, zero.imag]), *simplex_points
                    ):
                        zeros.append(zero)
                        aaa_z_minus_p += 1
            else:
                raise ValueError("Invalid option for `approx_func`.")

            # debug plotting, can remove later
            if plot_aaa:
                fig, ax = plt.subplots()
                phase_plot(
                    aaa_f,
                    ax,
                    domain=[
                        np.min(tri.points[:, 0]),
                        np.max(tri.points[:, 0]),
                        np.min(tri.points[:, 1]),
                        np.max(tri.points[:, 1]),
                    ],
                )
                plot_poles_zeros(aaa_f, ax)
                ax.plot(z.real, z.imag, ".")
                plt.show()

            # report if AAA and argument principle are not matching and destroy the
            # triangle if so
            if not np.allclose(arg_princ_z_minus_p, aaa_z_minus_p):
                print(
                    f"AAA and argument principle don't match: {aaa_z_minus_p=}, {arg_princ_z_minus_p=}"
                )
                refine_further = True
                new_points = np.concatenate(
                    [new_points, compute_incenter(*simplex_points)[np.newaxis]]
                )

        if not refine_further:
            return ZerosPolesResult(
                zeros=np.asarray(zeros),
                poles=np.asarray(poles),
                residues=np.asarray(residues),
                points=tri.points,
                simplices=tri.simplices,
            )
        else:
            points = convert_cart_to_complex(new_points)


if __name__ == "__main__":
    from poles_roots import reference_problems

    find_zeros_poles(
        reference_problems.func0,
        reference_problems.func0_jac,
        initial_points=[-10 - 10j, 10 - 10j, 10 + 10j, -10 + 10j],
        arg_principal_threshold=4.1,
        num_sample_points=100,
    )
