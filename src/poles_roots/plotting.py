import numpy as np

from poles_roots.aaa import AAA

from colorsys import hls_to_rgb


def colorize(z):
    r = np.abs(z)
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 0.5  # 1.0 - 1.0 / (1.0 + (r/r.max())**0.005)
    s = 0.8

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.transpose(1, 2, 0)
    return c


def phase_plot(f, ax, /, *, domain=None, classic=False, n_points=500):
    """Plots the complex phase plane of `f`."""
    theta = -np.pi
    domain = [-1, 1, -1, 1] if domain is None else domain

    x = np.linspace(domain[0], domain[1], num=n_points)
    y = np.linspace(domain[2], domain[3], num=n_points)

    [xx, yy] = np.meshgrid(x, y)
    zz = xx + yy * 1j

    im = ax.pcolormesh(
        np.real(zz),
        np.imag(zz),
        colorize(f(zz)),
        shading="gouraud",
        cmap="twilight_shifted",
    )
    ax.set_xlabel(r"$\Re$(z)")
    ax.set_ylabel(r"$\Im$(z)")
    ax.set_xlim(domain[:2])
    ax.set_ylim(domain[2:])
    return im


def plot_poles_zeros(
    result_object: AAA,
    ax,
    /,
    *,
    expected_poles=None,
    expected_zeros=None,
):
    """Adds the poles and zeros computed by AAA to a plot."""
    ax.plot(
        np.real(result_object.poles),
        np.imag(result_object.poles),
        "rx",
        markersize=10,
        label="Computed Poles",
    )
    ax.plot(
        np.real(result_object.roots),
        np.imag(result_object.roots),
        "gx",
        markersize=10,
        label="Computed Zeros",
    )
    if expected_poles is not None:
        ax.plot(
            np.real(expected_poles),
            np.imag(expected_poles),
            "ro",
            label="Expected Poles",
            mfc="none",
            markersize=13,
        )
    if expected_zeros is not None:
        ax.plot(
            np.real(expected_zeros),
            np.imag(expected_zeros),
            "go",
            label="Expected Poles",
            mfc="none",
            markersize=13,
        )


def plot_triangulation_with_argument_principle(
    f,
    ax,
    points,
    simplices,
    z_minus_p,
    to_insert=None,
    formatting=".1E",
):
    """Plots a triangulation with with the argument principle result in the centre of
    each simplex."""
    phase_plot(f, ax, domain=[-10, 10, -10, 10])
    ax.triplot(points[:, 0], points[:, 1], simplices)
    ax.plot(points[:, 0], points[:, 1], "o")

    for i, simplex in enumerate(simplices):
        if to_insert is None:
            color = "g"
        else:
            color = "r" if to_insert[i] else "g"

        centre = points[simplex, :].mean(axis=0)
        ax.text(
            centre[0],
            centre[1],
            f"{np.real(z_minus_p[i]):{formatting}}",
            color=color,
        )
