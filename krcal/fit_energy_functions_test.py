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
from invisible_cities.icaro. hst_functions     import shift_to_bin_centers

from invisible_cities.core .stat_functions     import poisson_sigma
from   invisible_cities.core.core_functions    import in_range

from . core_functions       import gaussian_parameters
from . core_functions       import find_nearest
from . core_functions       import divide_np_arrays
from . fit_functions        import chi2
from . fit_energy_functions import energy_fit


def test_energy_fit():
    Nevt  = int(1e6)
    mean = 100
    sigma = 10
    bins  = 100
    range = (0,200)
    n_sigma =3
    e = np.random.normal(mean, sigma, Nevt)

    fc =energy_fit (e , bins, range, n_sigma =n_sigma)

    hp = fc.hp
    fp = fc.fp
    seed = fc.seed

    gp = gaussian_parameters(e, range)

    assert seed.mu   == approx(gp.mu  , rel=1e-3)
    assert seed.std  == approx(gp.std , rel=1e-3)
    assert seed.amp  == approx(gp.amp , rel=1e-3)

    y, b = np.histogram(e, bins= bins, range=range)
    x = shift_to_bin_centers(b)

    fit_range = seed.mu - n_sigma * seed.std, seed.mu + n_sigma * seed.std

    x, y      = x[in_range(x, *fit_range)], y[in_range(x, *fit_range)]

    assert_array_equal(x, fp.x)
    assert_array_equal(y, fp.y)
    assert_array_equal(poisson_sigma(y), fp.yu)

    assert chi2(fp.f, fp.x, fp.y, fp.yu) == approx(fp.chi2, rel=1e-3)
    ok = fp.valid
    assert ok == True 

    assert_array_equal(e, hp.var)
    assert bins  == approx(hp.nbins , rel=1e-3)
    assert range[0]  == approx(hp.range[0] , rel=1e-3)
    assert range[1]  == approx(hp.range[1] , rel=1e-3)




# @given(floats(min_value = 100,
#               max_value = +1000),
#        floats(min_value = + 1,
#               max_value = + 20))
# @settings(max_examples=10)
#
# def test_mean_and_std_positive(mean, sigma):
#     Nevt  = int(1e6)
#     e = np.random.normal(mean, sigma, Nevt)
#
#     mu, std = mean_and_std(e, (mean- 3 * sigma,mean + 3 * sigma))
#     assert mu   == approx(mean  , rel=1e-1)
#     assert std  == approx(sigma,  rel=1e-1)
#
#
#
# @given(floats(min_value = -100,
#               max_value = +100),
#        floats(min_value = + 0.1,
#               max_value = + 10))
# @settings(max_examples=10)
#
# def test_mean_and_std_zero(mean, sigma):
#     Nevt  = int(1e6)
#     e = np.random.normal(mean, sigma, Nevt)
#
#     mu, std = mean_and_std(e, (mean- 3 * sigma,mean + 3 * sigma))
#     assert mu   == approx(mean  , abs=1e-1)
#     assert std  == approx(sigma,  abs=1e-1)
#
#
# def test_simple_find_nearest():
#     x = np.arange(100)
#     assert find_nearest(x, 75.6)   == exactly(76)
#     assert find_nearest(x, 75.5)   == exactly(75)
#
#
# def test_gauss_find_nearest():
#     e = np.random.normal(100, 10, 100)
#
#     for x in range(1, 100, 10):
#         assert find_nearest(e, x)   == approx(nearest(e, x), rel=1e-3)
#
#
# @given(float_arrays(min_value=1,
#                     max_value=100))
# def test_find_nearest(data):
#     assert find_nearest(data, 10)   == approx(nearest(data, 10), rel=1e-3)
#
#
# def test_divide_np_array():
#     x = np.array([10,100,1000,5000])
#     y = np.array([2,5,0,100])
#     assert_array_equal(divide_np_arrays(x,y), np.array([  5.,  20.,   0.,  50.]))
