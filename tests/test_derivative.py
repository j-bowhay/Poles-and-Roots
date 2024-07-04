import pytest
from numpy.testing import assert_allclose
import numpy as np

from poles_roots._diff import complex_derivitive


@pytest.mark.parametrize("f,z,expected", [(np.sin, 0, np.cos)])
def test_complex_derivitive(f, z, expected):
    assert_allclose(complex_derivitive(f, z, 100), expected(z))
