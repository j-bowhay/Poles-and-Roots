import numpy as np
from numpy.testing import assert_equal

from poles_roots.utils import convert_cart_to_complex


def test_convert_cart_to_complex():
    points = np.array([[1, 1], [10.7, -8], [-99.8, 7]])
    z = convert_cart_to_complex(points)
    assert_equal(z, [1 + 1j, 10.7 - 8j, -99.8 + 7j])
