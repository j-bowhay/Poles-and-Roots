from typing import Optional
import numpy as np
import scipy
import matplotlib.pyplot as plt

from poles_roots.integration import argument_priciple_of_triangulation
from poles_roots.plotting import plot_triangulation_with_argument_principle
from poles_roots._utils import compute_incenter


def adaptive_triangulation(
    f,
    f_jac,
    initial_points,
    arg_principal_threshold,
    plot=False,
    quad_kwargs: Optional[dict] = None,
):
    initial_points = np.asarray(initial_points)
    points = np.column_stack([initial_points.real, initial_points.imag])
    tri = scipy.spatial.Delaunay(points)

    while np.any(
        (
            to_insert := np.abs(
                (
                    z_minus_p := argument_priciple_of_triangulation(
                        f,
                        f_jac,
                        tri.points,
                        tri.simplices,
                        quad_kwargs,
                    )[0]
                ),
            )
            > arg_principal_threshold
        ),
    ):
        if plot:
            fig, ax = plt.subplots()
            plot_triangulation_with_argument_principle(
                f,
                ax,
                tri.points,
                tri.simplices,
                z_minus_p,
                to_insert,
            )
            plt.show()

        # compute the points to be added to the triangulation
        insert_index = np.nonzero(to_insert)[0]
        points_to_add = np.empty((insert_index.size, 2))
        for i, simplex in enumerate(tri.simplices[insert_index]):
            A, B, C = tri.points[simplex, :]
            points_to_add[i, :] = compute_incenter(A, B, C)
        points = np.concatenate([points, points_to_add])
        tri = scipy.spatial.Delaunay(points)

    if plot:
        fig, ax = plt.subplots()
        plot_triangulation_with_argument_principle(
            f,
            ax,
            tri.points,
            tri.simplices,
            z_minus_p,
        )
        ax.set_title("Final Triangulation")
        plt.show()

    return tri.points, tri.simplices, z_minus_p


if __name__ == "__main__":
    adaptive_triangulation(
        lambda z: 1 / z,
        lambda z: -1 / z**2,
        [-5 - 5j, 5 - 5j, 5 + 5j, -5 + 5j],
        arg_principal_threshold=1.1,
        plot=True,
    )
