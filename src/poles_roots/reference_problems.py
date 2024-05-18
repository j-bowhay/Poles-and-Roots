import numpy as np


def func0(z):
    out = 1 / (z - 0.5)
    out_jac = -1 / (z - 0.5) ** 2

    return out, out_jac


def func1(z):
    out = z - 2
    out_jac = 1

    return out, out_jac


def func2(z):
    out = np.exp(3 * z) + 2 * z * np.cos(z) - 1
    out_jac = 3 * np.exp(3 * z) + 2 * (np.cos(z) - z * np.sin(z))

    return out, out_jac


def func3(z):
    out = z * z * (z - 1) * (z - 2) * (z - 3) * (z - 4) + z * np.sin(z)
    out_jac = (
        2 * z * (3 * z * z * z * z - 25 * z * z * z + 70 * z * z - 75 * z + 24)
        + np.sin(z)
        + z * np.cos(z)
    )

    return out, out_jac


def func4(z):
    out = (
        z
        * z
        * np.square(z - 2)
        * (z * z * z + np.exp(2 * z) * np.cos(z) - np.sin(z) - 1)
    )
    out_jac = z * (z - 2) * (
        7 * z * z * z * z
        - 10 * z * z * z
        - 4 * z
        + 4
        + (2 * (z * z - 2) * np.exp(2 * z) - z * (z - 2)) * np.cos(z)
    ) + (-z * (z - 2) * np.exp(2 * z) - 4 * z + 4) * np.sin(z)

    return out, out_jac


def func5(z):
    out = (
        (z - 1)
        * (z - 2)
        * (z - 3)
        * (z - 4)
        * (z - 5)
        * (z - 6)
        * (z - 7)
        * (z - 8)
        * (z - 9)
        * (z - 10)
    )
    out_jac = (2 * z - 11) * (
        966240
        + (z - 11)
        * z
        * (194832 + (z - 11) * z * (14124 + 5 * (z - 11) * z * (88 + (z - 11) * z)))
    )

    return out, out_jac
