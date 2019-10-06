"""
Tests for fit_functions
"""

import numpy as np

from numpy.testing import assert_allclose


from invisible_cities.core.testing_utils       import exactly

import matplotlib.pyplot as plt
from . histo_functions              import h1

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


def test_h1():
    Nevt  = int(1e5)
    mean = 200
    sigma = 10
    nbins =50
    x = np.random.normal(mean, sigma, Nevt)
    r = mean - 5 * sigma, mean + 5 * sigma
    n, b, _, _ = h1(x, bins=nbins, range=r)
    imax = np.argmax(n)
    assert len(n)   == exactly(nbins)
    assert b[imax - 2] < mean <  b[imax + 2]
