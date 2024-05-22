import numpy as np


def func0(z):
    return 1 / (z - 0.5)


def func0_jac(z):
    return -1 / (z - 0.5) ** 2


def func1(z):
    return z - 2


def func1_jac(z):
    return 1


def func2(z):
    return np.exp(3 * z) + 2 * z * np.cos(z) - 1


def func2_jac(z):
    return 3 * np.exp(3 * z) + 2 * (np.cos(z) - z * np.sin(z))


def func3(z):
    return z * z * (z - 1) * (z - 2) * (z - 3) * (z - 4) + z * np.sin(z)


def func3_jac(z):
    return (
        2 * z * (3 * z * z * z * z - 25 * z * z * z + 70 * z * z - 75 * z + 24)
        + np.sin(z)
        + z * np.cos(z)
    )


def func4(z):
    return (
        z
        * z
        * np.square(z - 2)
        * (z * z * z + np.exp(2 * z) * np.cos(z) - np.sin(z) - 1)
    )


def func4_jac(z):
    return z * (z - 2) * (
        7 * z * z * z * z
        - 10 * z * z * z
        - 4 * z
        + 4
        + (2 * (z * z - 2) * np.exp(2 * z) - z * (z - 2)) * np.cos(z)
    ) + (-z * (z - 2) * np.exp(2 * z) - 4 * z + 4) * np.sin(z)


def func5(z):
    return (
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


def func5_jac(z):
    return (2 * z - 11) * (
        966240
        + (z - 11)
        * z
        * (194832 + (z - 11) * z * (14124 + 5 * (z - 11) * z * (88 + (z - 11) * z)))
    )
