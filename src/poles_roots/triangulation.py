from typing import Optional
import numpy as np
import scipy
import matplotlib.pyplot as plt

from poles_roots.integration import argument_priciple_of_triangulation
from poles_roots.plotting import phase_plot


def adaptive_triangulation(
    f,
    f_jac,
    initial_points,
    arg_pricipal_threshold,
    plot=False,
    quad_kwargs: Optional[dict] = None,
):
    quad_kwargs = {} if quad_kwargs is None else quad_kwargs
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
            > arg_pricipal_threshold
        ),
    ):
        if plot:
            fig, ax = plt.subplots()
            phase_plot(f, ax, domain=[-10, 10, -10, 10])
            ax.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices)
            ax.plot(tri.points[:, 0], tri.points[:, 1], "o")

            for i, simplex in enumerate(tri.simplices):
                color = "r" if to_insert[i] else "g"
                centre = tri.points[simplex, :].mean(axis=0)
                ax.text(
                    centre[0],
                    centre[1],
                    f"{np.real(z_minus_p[i]):.1E}",
                    color=color,
                )
            plt.show()

        tri.add_points(
            tri.points[tri.simplices[np.nonzero(to_insert)[0]], :].mean(axis=1),
        )

    tri.close()


if __name__ == "__main__":
    from poles_roots.reference_problems import func2, func2_jac

    adaptive_triangulation(
        func2,
        func2_jac,
        [-9 - 9.2j, 9 - 10j, 9.9 + 9.8j, -9.3 + 9.5j],
        arg_pricipal_threshold=1.1,
        plot=True,
    )
