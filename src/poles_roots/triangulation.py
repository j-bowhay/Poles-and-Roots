from typing import Optional
import numpy as np
import scipy
import matplotlib.pyplot as plt

from poles_roots.integration import argument_priciple_of_triangulation
from poles_roots.plotting import plot_triangulation_with_argument_principle
from poles_roots.utils import compute_incenter


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
    tri = scipy.spatial.Delaunay(points, incremental=True)

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
                    )
                ),
            )
            > arg_principal_threshold
        ),
    ):
        if plot:
            plot_triangulation_with_argument_principle(
                f,
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
        tri.add_points(points_to_add)

    tri.close()
    if plot:
        _, ax = plot_triangulation_with_argument_principle(
            f,
            tri.points,
            tri.simplices,
            z_minus_p,
        )
        ax.set_title("Final Triangulation")
        plt.show()

    return tri.points, tri.simplices


if __name__ == "__main__":
    from poles_roots.reference_problems import func2, func2_jac

    adaptive_triangulation(
        func2,
        func2_jac,
        [-9 - 9.2j, 9 - 10j, 9.9 + 9.8j, -9.3 + 9.5j],
        arg_principal_threshold=1.1,
        plot=True,
    )
