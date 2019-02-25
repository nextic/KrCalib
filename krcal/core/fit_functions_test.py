"""
Tests for fit_functions
"""

import numpy as np

from pytest                import approx

from hypothesis            import given
from hypothesis.strategies import floats

from . fit_functions       import sigmoid


@given(floats(min_value = -1e4,
              max_value = +1e4),
       floats(min_value = -1e4,
              max_value = + 1e4))
def test_sigmoid_limits(A, D):
    aux_sigmoid = lambda x: sigmoid(x, A, 0, 10, D)
    assert aux_sigmoid(-10) == approx(D  , rel=1e-2)
    assert aux_sigmoid( 10) == approx(A+D, rel=1e-2)

@given(floats(min_value = -1e4,
              max_value = +1e4),
       floats(min_value = -1e4,
              max_value = + 1e4),
       floats(min_value = -1e2,
              max_value = + 1e2),
       floats(min_value = -1e4,
              max_value = + 1e4))
def test_sigmoid_values_at_abscissa_axis(A, B, C, D):
    value = sigmoid(0, A, B, C, D)
    test  = A / (1 + np.exp(B*C)) + D
    assert value == test
