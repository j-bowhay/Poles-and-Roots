import numpy as np
from matplotlib import cm
from matplotlib import colors
from poles_roots.aaa import AAA


def colorize(z):
    s1 = np.mod(np.log(np.abs(z)), 1)
    s2 = np.mod(np.angle(z), 2 * np.pi) / (2 * np.pi)
    col = colors.rgb_to_hsv(cm.hsv(s2)[:, :, :3])
    x = 0.6 + 0.4 * s1
    col[:, :, 2] *= x
    return colors.hsv_to_rgb(col)


def phase_plot(f, ax, /, *, domain=None, classic=False, n_points=1000):
    """Plots the complex phase plane of `f`."""
    domain = [-1, 1, -1, 1] if domain is None else domain

    x = np.linspace(domain[0], domain[1], num=n_points)
    y = np.linspace(domain[2], domain[3], num=n_points)

    [xx, yy] = np.meshgrid(x, y)
    zz = xx + yy * 1j

    im = ax.imshow(colorize(f(zz)), extent=domain, origin="lower", aspect="auto")
    ax.set_xlabel(r"$\Re$(z)")
    ax.set_ylabel(r"$\Im$(z)")
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
