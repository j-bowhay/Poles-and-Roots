import pytest
from numpy.testing import assert_allclose

from poles_roots.global_zero_pole import find_zeros_poles
from poles_roots import reference_problems


@pytest.mark.parametrize(
    "f,f_prime,arg_principal_threshold,expected_pole,expected_zeros",
    [
        (reference_problems.func0, reference_problems.func0_jac, 1.1, [0.5], []),
        (reference_problems.func1, reference_problems.func1_jac, 1.1, [], [2]),
    ],
)
def test_find_zeros_poles(
    f, f_prime, arg_principal_threshold, expected_pole, expected_zeros
):
    res = find_zeros_poles(
        f,
        f_prime,
        initial_points=[-10 - 10j, 10 - 10j, 10 + 10j, -10 + 10j],
        arg_principal_threshold=arg_principal_threshold,
        num_sample_points=50,
    )
    assert_allclose(res.poles, expected_pole)
    assert_allclose(res.zeros, expected_zeros)
