import pytest
from numpy.testing import assert_allclose
import numpy as np

from poles_roots.kravanja_van_barel import kravanja_van_barel


@pytest.mark.parametrize(
    "f,f_prime,N,expected_zeros,expected_multiplicities",
    [
        (lambda z: z, lambda z: 1, 1, [0], [1]),
        (lambda z: z, lambda z: 1, None, [0], [1]),
        (
            lambda z: (z + 0.5) * (z - 0.5) * z,
            lambda z: 3 * z**2 - 0.25,
            3,
            [0.5, -0.5, 0],
            [1, 1, 1],
        ),
        (
            lambda z: (z + 0.5) * (z - 0.5) * z,
            lambda z: 3 * z**2 - 0.25,
            None,
            [0.5, -0.5, 0],
            [1, 1, 1],
        ),
        (lambda z: (z + 0.5j) * (z - 0.5j), lambda z: 2 * z, 2, [0.5j, -0.5j], [1, 1]),
        (lambda z: (z + 0.5) * (z - 0.5), lambda z: 2 * z, 2, [0.5, -0.5], [1, 1]),
        (lambda z: (z + 0.5) * (z - 0.5), lambda z: 2 * z, None, [0.5, -0.5], [1, 1]),
        (lambda z: z**2 * (z - 0.5), lambda z: z * (3 * z - 1), None, [0, 0.5], [2, 1]),
    ],
)
def test_delves_lyness(f, f_prime, N, expected_zeros, expected_multiplicities):
    points = np.array([[-1 - 1j], [1 - 1j], [1 + 1j], [-1 + 1j]])
    zeros, multiplicities = kravanja_van_barel(f, f_prime, points, N)
    assert_allclose(np.sort(zeros), np.sort(expected_zeros), atol=1e-9)
    assert_allclose(np.sort(multiplicities), np.sort(multiplicities))
