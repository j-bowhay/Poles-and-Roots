from typing import Optional
import numpy as np
import scipy
import matplotlib.pyplot as plt

from poles_roots.integration import argument_priciple_of_triangulation
from poles_roots.plotting import plot_triangulation_with_argument_principle
from poles_roots._utils import compute_incenter, point_in_polygon


def adaptive_triangulation(
    f,
    f_jac,
    initial_points,
    arg_principal_threshold,
    plot=False,
    quad_kwargs: Optional[dict] = None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    initial_points = np.asarray(initial_points)
    points = np.column_stack([initial_points.real, initial_points.imag])
    tri = scipy.spatial.Delaunay(points)
    z_minus_p, to_destroy = argument_priciple_of_triangulation(
        f,
        f_jac,
        tri.points,
        tri.simplices,
        quad_kwargs,
    )

    while (
        np.any(to_insert := np.abs(z_minus_p) > arg_principal_threshold) or to_destroy
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

        if to_destroy:
            # destroy edge if we are integrating through a pole
            points_to_add = np.empty((len(to_destroy), 2))
            for i, (a, b) in enumerate(to_destroy):
                hull = scipy.spatial.ConvexHull(tri.points)
                # before doing anything check the edge to destroy is in the convex hull
                edge = np.array([[a.real, a.imag], [b.real, b.imag]])
                hull_points = tri.points[hull.vertices, :]
                for j, first_point in enumerate(hull_points):
                    second_point = np.take(hull_points, j + 1, axis=0, mode="wrap")
                    if np.all(edge == np.stack([first_point, second_point])) or np.all(
                        edge == np.stack([second_point, first_point])
                    ):
                        raise ValueError("Pole/Zero detected on the convex hull.")

                acceptable_point = False
                while not acceptable_point:
                    center = (a + b) / 2
                    radius = np.abs(a - b) / 2
                    r = radius * np.sqrt(rng.random())
                    theta = rng.uniform(0, 2 * np.pi)
                    new = center + r * np.exp(theta * 1j)
                    point = [new.real, new.imag]
                    if point_in_polygon(point, tri.points[hull.vertices, :]):
                        acceptable_point = True
                points_to_add[i, :] = point
        else:
            # compute the points to be added to the triangulation
            insert_index = np.flatnonzero(to_insert)
            points_to_add = np.empty((insert_index.size, 2))
            # destroy triangles where |Z-P| is too large
            for i, simplex in enumerate(tri.simplices[insert_index]):
                A, B, C = tri.points[simplex, :]
                points_to_add[i, :] = compute_incenter(A, B, C)

        # add new points to triangulation
        points = np.concatenate([points, points_to_add])
        tri = scipy.spatial.Delaunay(points)

        # recompute the argument principle
        z_minus_p, to_destroy = argument_priciple_of_triangulation(
            f,
            f_jac,
            tri.points,
            tri.simplices,
            quad_kwargs,
        )

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
        lambda z: z,
        lambda z: 1,
        [-5, 5, 5 - 5j, -5 - 5j],
        arg_principal_threshold=1.1,
        plot=True,
    )
