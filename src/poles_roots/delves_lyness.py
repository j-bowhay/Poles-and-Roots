import numpy as np

from poles_roots.integration import argument_principle_from_points


def delves_lyness(f, f_prime, points, N=None):
    if N is None:
        N = int(argument_principle_from_points(f, f_prime, points, moment=0)[0].real)
    s = np.empty(N, dtype=np.complex128)

    for i in range(N):
        s[i] = argument_principle_from_points(f, f_prime, points, moment=i + 1)[0]

    coeffs = np.ones(N + 1)

    A = np.zeros((N, N))
    coeffs[1:] = np.linalg.solve(A, -s)

    return np.roots(coeffs)


if __name__ == "__main__":
    delves_lyness(
        lambda z: (z - 0.5) * (z + 0.5),
        lambda z: 2 * z,
        [-1 - 1j, 1 - 1j, 1 + 1j, -1 + 1j],
    )
