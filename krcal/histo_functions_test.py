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

import matplotlib.pyplot as plt
from . histo_functions              import h1d
from invisible_cities.icaro. hst_functions import shift_to_bin_centers

def test_matplotlib_histo_equals_numpy_histo():
    Nevt  = int(1e5)
    mean = 100
    sigma = 10
    x = np.random.normal(mean, sigma, Nevt)
    hist, bins, p = plt.hist(x,
                         bins= 100,
                         range=(0,200))

    hist2, bins2  = np.histogram(x,
                         bins= 100,
                         range=(0,200))

    assert_allclose(hist, hist2)
    assert_allclose(bins, bins2)


def test_h1d():
    Nevt  = int(1e5)
    mean = 200
    sigma = 10
    nbins =50
    x = np.random.normal(mean, sigma, Nevt)
    r = mean - 5 * sigma, mean + 5 * sigma
    n, b, p = h1d(x, bins=nbins, range=r)
    imax = np.argmax(n)
    assert len(n)   == exactly(nbins)
    assert b[imax - 2] < mean <  b[imax + 2]
