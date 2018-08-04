"""
Tests for stat_functions
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

from . stat_functions       import gaussian_experiment
from . stat_functions       import mean_and_std


def test_simple_mean_and_std():
    Nevt  = 1e6
    mean = 100
    sigma = 10
    e = gaussian_experiment(nevt=Nevt, mean=mean, std=sigma)
    mu, std = mean_and_std(e, (0,200))
    assert mu   == approx(100  , rel=1e-2)
    assert std  == approx(10, rel=1e-2)


@given(floats(min_value = 100,
              max_value = +1000),
       floats(min_value = + 1,
              max_value = + 20))
@settings(max_examples=50)

def test_mean_and_std_positive(mean, sigma):
    Nevt  = int(1e5)
    e = gaussian_experiment(nevt=Nevt, mean=mean, std=sigma)

    mu, std = mean_and_std(e, (mean- 5 * sigma,mean + 5 * sigma))
    assert mu   == approx(mean  , rel=1e-2)
    assert std  == approx(sigma,  rel=1e-2)


@given(floats(min_value = -100,
              max_value = +100),
       floats(min_value = + 0.1,
              max_value = + 10))
@settings(max_examples=10)

def test_mean_and_std_zero(mean, sigma):
    Nevt  = int(1e5)
    e = gaussian_experiment(nevt=Nevt, mean=mean, std=sigma)

    mu, std = mean_and_std(e, (mean- 5 * sigma,mean + 5 * sigma))
    assert mu   == approx(mean  , abs=1e-1)
    assert std  == approx(sigma,  abs=1e-1)
