"""
Tests for fit_functions
"""

import numpy as np
from   numpy.testing        import assert_array_equal
from   numpy.testing        import assert_equal
from   numpy.testing        import assert_allclose

from  . core_functions import divide_np_arrays
from  . core_functions import resolution

def test_divide_np_array():
    x = np.array([10,100,1000,5000])
    y = np.array([2,5,0,100])
    assert_array_equal(divide_np_arrays(x,y), np.array([  5.,  20.,   0.,  50.]))


def test_resolution_no_errors():
    R, Rbb = resolution([None, 1, 1])

    assert_equal(R  .uncertainty, 0)
    assert_equal(Rbb.uncertainty, 0)


def test_resolution_scaling():
    _, Rbb1 = resolution([None, 1, 1], E_from = 1)
    _, Rbb2 = resolution([None, 1, 1], E_from = 2)

    assert_allclose(Rbb1.value * 2**0.5, Rbb2.value)
