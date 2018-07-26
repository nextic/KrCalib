"""
Tests for fit_functions
"""

import numpy as np

from pytest        import mark
from pytest        import approx
from pytest        import raises
from flaky         import flaky
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from hypothesis            import given, settings
from hypothesis.strategies import integers
from hypothesis.strategies import floats
from invisible_cities.core.testing_utils       import exactly
from invisible_cities.core.testing_utils       import float_arrays
from invisible_cities.core.testing_utils       import FLOAT_ARRAY
from invisible_cities.core.testing_utils       import random_length_float_arrays

from . core_functions       import mean_and_std
from . core_functions       import find_nearest
from . core_functions       import divide_np_arrays


def nearest(a, v):
    """Alternative (not optimized) implementation of find_nearest
    Used for testing purpose only

    """
    nr =a[0]
    diff = 1e+9
    for x in a:
        if np.abs(x-v) < diff:
            nr = x
            diff = np.abs(x-v)
    return nr

def test_simple_mean_and_std():
    Nevt  = int(1e6)
    mean = 100
    sigma = 10
    e = np.random.normal(mean, sigma, Nevt)
    mu, std = mean_and_std(e, (0,200))
    assert mu   == approx(100  , rel=1e-2)
    assert std  == approx(10, rel=1e-2)


@given(floats(min_value = 100,
              max_value = +1000),
       floats(min_value = + 1,
              max_value = + 20))
@settings(max_examples=10)

def test_mean_and_std_positive(mean, sigma):
    Nevt  = int(1e6)
    e = np.random.normal(mean, sigma, Nevt)

    mu, std = mean_and_std(e, (mean- 3 * sigma,mean + 3 * sigma))
    assert mu   == approx(mean  , rel=1e-1)
    assert std  == approx(sigma,  rel=1e-1)



@given(floats(min_value = -100,
              max_value = +100),
       floats(min_value = + 0.1,
              max_value = + 10))
@settings(max_examples=10)

def test_mean_and_std_zero(mean, sigma):
    Nevt  = int(1e6)
    e = np.random.normal(mean, sigma, Nevt)

    mu, std = mean_and_std(e, (mean- 3 * sigma,mean + 3 * sigma))
    assert mu   == approx(mean  , abs=1e-1)
    assert std  == approx(sigma,  abs=1e-1)


def test_simple_find_nearest():
    x = np.arange(100)
    assert find_nearest(x, 75.6)   == exactly(76)
    assert find_nearest(x, 75.5)   == exactly(75)


def test_gauss_find_nearest():
    e = np.random.normal(100, 10, 100)

    for x in range(1, 100, 10):
        assert find_nearest(e, x)   == approx(nearest(e, x), rel=1e-3)


@given(float_arrays(min_value=1,
                    max_value=100))
def test_find_nearest(data):
    assert find_nearest(data, 10)   == approx(nearest(data, 10), rel=1e-3)


def test_divide_np_array():
    x = np.array([10,100,1000,5000])
    y = np.array([2,5,0,100])
    assert_array_equal(divide_np_arrays(x,y), np.array([  5.,  20.,   0.,  50.]))
