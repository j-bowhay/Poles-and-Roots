import numpy as np

from poles_roots.aaa import _AAAResult


def phase_plot(f, ax, /, *, domain=None, classic=False, n_points=500):
    theta = -np.pi
    domain = [-1, 1, -1, 1] if domain is None else domain

    x = np.linspace(domain[0], domain[1], num=n_points)
    y = np.linspace(domain[2], domain[3], num=n_points)

    [xx, yy] = np.meshgrid(x, y)
    zz = xx + yy * 1j

    if classic:

        def phi(t):
            return t
    else:

        def phi(t):
            return t - 0.5 * np.cos(1.5 * t) ** 3 * np.sin(1.5 * t)

    angle = np.mod(phi(np.angle(f(zz)) - np.pi) + np.pi - theta, 2 * np.pi) + theta
    im = ax.pcolormesh(
        np.real(zz),
        np.imag(zz),
        angle,
        shading="gouraud",
        cmap="twilight_shifted",
    )
    ax.set_xlabel(r"$\Re$(z)")
    ax.set_ylabel(r"$\Im$(z)")
    ax.set_xlim(domain[:2])
    ax.set_ylim(domain[2:])
    return im


def plot_poles_zeros(
    result_object: _AAAResult,
    ax,
    /,
    *,
    expected_poles=None,
    expected_zeros=None,
):
    ax.plot(
        np.real(result_object.poles),
        np.imag(result_object.poles),
        "rx",
        markersize=8,
        label="Computed Poles",
    )
    ax.plot(
        np.real(result_object.zeros),
        np.imag(result_object.zeros),
        "gx",
        markersize=8,
        label="Computed Zeros",
    )
    if expected_poles is not None:
        ax.plot(
            np.real(expected_poles),
            np.imag(expected_poles),
            "ro",
            label="Expected Poles",
            mfc="none",
            markersize=11,
        )
    if expected_zeros is not None:
        ax.plot(
            np.real(expected_zeros),
            np.imag(expected_zeros),
            "go",
            label="Expected Poles",
            mfc="none",
            markersize=11,
        )
