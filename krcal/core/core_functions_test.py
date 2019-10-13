"""
Tests for fit_functions
"""

import numpy as np
from   numpy.testing        import assert_array_equal
from   .     core_functions import divide_np_arrays

def test_divide_np_array():
    x = np.array([10,100,1000,5000])
    y = np.array([2,5,0,100])
    assert_array_equal(divide_np_arrays(x,y), np.array([  5.,  20.,   0.,  50.]))
