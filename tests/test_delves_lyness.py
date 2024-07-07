import pytest
from numpy.testing import assert_allclose
import numpy as np

from poles_roots.delves_lyness import delves_lyness


@pytest.mark.parametrize(
    "f,f_prime,N,zeros",
    [
        (lambda z: z, lambda z: 1, 1, [0]),
        (lambda z: z, lambda z: 1, None, [0]),
        (
            lambda z: (z + 0.5) * (z - 0.5) * z,
            lambda z: 3 * z**2 - 0.25,
            3,
            [0.5, -0.5, 0],
        ),
        (
            lambda z: (z + 0.5) * (z - 0.5) * z,
            lambda z: 3 * z**2 - 0.25,
            None,
            [0.5, -0.5, 0],
        ),
        (lambda z: (z + 0.5j) * (z - 0.5j), lambda z: 2 * z, 2, [0.5j, -0.5j]),
        (lambda z: (z + 0.5) * (z - 0.5), lambda z: 2 * z, 2, [0.5, -0.5]),
        (
            lambda z: (z + 0.5) * (z - 0.5),
            lambda z: 2 * z,
            None,
            [0.5, -0.5],
        ),
    ],
)
def test_delves_lyness(f, f_prime, N, zeros):
    points = np.array([[-1 - 1j], [1 - 1j], [1 + 1j], [-1 + 1j]])
    res = delves_lyness(f, f_prime, points, N)
    assert_allclose(np.sort(res), np.sort(zeros))
