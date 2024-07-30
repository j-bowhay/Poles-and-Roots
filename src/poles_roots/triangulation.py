from typing import Optional
import numpy as np
import scipy
import matplotlib.pyplot as plt

from poles_roots.integration import argument_priciple_of_triangulation
from poles_roots.plotting import plot_triangulation_with_argument_principle
from poles_roots._utils import point_in_polygon


def adaptive_triangulation(
    f,
    f_prime,
    initial_points,
    arg_principal_threshold,
    plot=False,
    quad_kwargs: Optional[dict] = None,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    initial_points = np.asarray(initial_points)
    # convert from complex to cartesian
    points = np.column_stack([initial_points.real, initial_points.imag])
    tri = scipy.spatial.Delaunay(points)
    z_minus_p, to_destroy = argument_priciple_of_triangulation(
        f,
        f_prime,
        tri.points,
        tri.simplices,
        quad_kwargs,
    )

    while (
        np.any(to_insert := np.abs(z_minus_p) > arg_principal_threshold) or to_destroy
    ):
        # debug plotting
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
            # destroy edge if we are integrating through a pole or a zero
            points_to_add = np.empty((len(to_destroy) * 2, 2))
            for i, (a, b) in enumerate(to_destroy):
                # We use scipy.spatial.Convex hull as the convex hull function in
                # scipy.spatial.Delaunay does not seem to guarantee anti-clockwise
                # ordering
                hull = scipy.spatial.ConvexHull(tri.points)
                # before doing anything check the edge to destroy is in the convex hull
                edge = np.array([[a.real, a.imag], [b.real, b.imag]])
                hull_points = tri.points[hull.vertices, :]
                for j, first_point in enumerate(hull_points):
                    second_point = np.take(hull_points, j + 1, axis=0, mode="wrap")
                    # need to check both directions as sets do not preserve order
                    if np.all(edge == np.stack([first_point, second_point])) or np.all(
                        edge == np.stack([second_point, first_point])
                    ):
                        raise ValueError("Pole/Zero detected on the convex hull.")

                # edge not on convex hull, proceed with destroying edge by inserting a
                # point in the circle whose diameter is between the to ends of the edge
                acceptable_points = np.zeros(2, dtype=np.bool_)
                points_for_edge = np.empty((2, 2))
                while np.any(~acceptable_points):
                    # for now we will just choose this point randomly, maybe there's
                    # a better way of doing this
                    center = (a + b) / 2
                    diff = a - b
                    radius = np.abs(diff) / 2
                    angle = np.angle(diff)
                    r = radius * np.sqrt(rng.random())
                    theta = rng.uniform(
                        [angle, angle + np.pi], [angle + np.pi, angle + 2 * np.pi]
                    )
                    new = (center + r * np.exp(theta * 1j))[~acceptable_points]
                    points_for_edge[~acceptable_points] = np.dstack(
                        [new.real, new.imag]
                    )
                    # check that the point we are inserting is within the search domain
                    for j, point in enumerate(points_for_edge):
                        if point_in_polygon(point, tri.points[hull.vertices, :]):
                            acceptable_points[j] = True
                points_to_add[2 * i : 2 * i + 2, :] = points_for_edge
        else:
            # no poles/zeros on any edges, we can proceed with destroy triangles
            # compute the points to be added to the triangulation
            insert_index = np.flatnonzero(to_insert)
            points_to_add = np.empty((insert_index.size, 2))
            # destroy triangles where |Z-P| is too large
            for i, simplex in enumerate(tri.simplices[insert_index]):
                # TODO: check that this point is in the convex hull
                points_to_add[i, :] = tri.points[simplex, :].mean(axis=0)

        # add new points to triangulation
        points = np.concatenate([points, points_to_add])
        tri = scipy.spatial.Delaunay(points)

        # recompute the argument principle
        z_minus_p, to_destroy = argument_priciple_of_triangulation(
            f,
            f_prime,
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

    return tri, z_minus_p


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from poles_roots.global_zero_pole import find_zeros_poles
    from poles_roots.plotting import phase_plot, plot_poles_zeros

    res = find_zeros_poles(
        lambda z: 1 / z,
        lambda z: -1 / z**2,
        initial_points=[-5 - 5j, 5 - 5j, 5 + 5j, -5 + 5j],
        arg_principal_threshold=1.1,
        num_sample_points=50,
    )

    fig, axs = plt.subplots(ncols=2, figsize=(12, 6))
    phase_plot(lambda z: 1 / z, axs[0], domain=[-6, 6, -6, 6])
    phase_plot(lambda z: 1 / z, axs[1], domain=[-6, 6, -6, 6])
    plot_poles_zeros(res, axs[0])
    axs[0].legend()
    axs[1].triplot(res.points[:, 0], res.points[:, 1], res.simplices)
    plt.show()
