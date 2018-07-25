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
from . core_functions       import mean_and_std
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



@given(floats(min_value = -100,
              max_value = +100),
       floats(min_value = + 0.1,
              max_value = + 10))
@settings(max_examples=10)

def test_fit_yields_expected_mean_std_and_chi2(mean, sigma):
    Nevt  = int(1e6)
    e = np.random.normal(mean, sigma, Nevt)
    bins  = 100
    n_sigma =3
    range = (mean - n_sigma * sigma, mean + n_sigma * sigma)

    mu, std = mean_and_std(e, (mean - n_sigma * sigma, mean + n_sigma * sigma))

    fc =energy_fit (e , bins, range, n_sigma = n_sigma)
    par  = np.array(fc.fp.f.values)

    mu_fit  = par[1]
    std_fit = par[2]
    assert mu         == approx(mu_fit,  abs=1e-1)
    assert std        == approx(std_fit,  abs=1e-1)
    assert fc.fp.chi2 == approx(1 , rel=0.3)
