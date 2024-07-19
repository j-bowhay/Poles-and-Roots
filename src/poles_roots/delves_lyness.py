import numpy as np
import scipy

from poles_roots.integration import argument_principle_from_points


def delves_lyness(f, f_prime, points, N=None):
    if N is None:
        N = round(argument_principle_from_points(f, f_prime, points, moment=0)[0].real)
    s = np.empty(N + 1, dtype=np.complex128)
    s[0] = N

    for i in range(1, N + 1):
        s[i] = argument_principle_from_points(f, f_prime, points, moment=i)[0]

    coeffs = np.ones(N + 1, dtype=np.complex128)

    A = scipy.linalg.toeplitz(s[:-1])
    np.fill_diagonal(A, np.arange(1, N + 1))
    A = np.tril(A)
    coeffs[1:] = np.linalg.solve(A, -s[1:])

    return np.roots(coeffs)
