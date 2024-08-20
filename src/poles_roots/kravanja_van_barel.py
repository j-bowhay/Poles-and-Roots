import numpy as np
import scipy

from poles_roots.integration import argument_principle_from_points


def kravanja_van_barel(f, f_prime, points, N=None, quad_kwargs=None):
    """Compute the zeros of `f` using the Kravanja and Van Barel method.

    Parameters
    ----------
    f : callable
        Function to compute the zeros of.
    f_prime : callable
        Derivative of `f`
    points : ndarray
        Points defining region
    N : int, optional
        Number of zeros in the region, by default None
    quad_kwargs : dict, optional
        Options to pass to integrator, by default None

    Returns
    -------
    array
        Computed zeros
    """
    if N is None:
        N = round(
            argument_principle_from_points(
                f, f_prime, points, moment=0, quad_kwargs=quad_kwargs
            )[0].real
        )

    s = np.empty(2 * N, dtype=np.complex128)
    s[0] = N

    for i in range(1, 2 * N):
        s[i] = argument_principle_from_points(
            f, f_prime, points, moment=i, quad_kwargs=quad_kwargs
        )[0]

    H_N = scipy.linalg.hankel(s[:N], s[N - 1 : -1])
    n = np.linalg.matrix_rank(H_N)

    H_N = scipy.linalg.hankel(s[:n], s[n - 1 : 2 * n - 1])
    H_N_reduced = scipy.linalg.hankel(s[1 : n + 1], s[n : 2 * n])

    zeros = scipy.linalg.eigvals(H_N_reduced, H_N)
    multiplicities = np.linalg.solve(np.vander(zeros, increasing=True).T, s[:n])
    return zeros, multiplicities
