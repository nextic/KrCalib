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
from invisible_cities.core.core_functions      import in_range
from invisible_cities.evm  .ic_containers      import Measurement

from . fit_energy_functions       import gaussian_parameters
from . fit_energy_functions       import energy_fit
from . fit_functions              import chi2
from . core_functions             import mean_and_std


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

    gp = gaussian_parameters(e,(mean - n_sigma * sigma, mean + n_sigma * sigma))
    fc  = energy_fit (e , bins, range, n_sigma = n_sigma)
    par  = fc.fr.par

    mu_fit  = par[1]
    std_fit = par[2]
    assert gp.mu.value   == approx(mu_fit,  abs=0.3)
    assert gp.std.value  == approx(std_fit,  abs=0.3)
    assert fc.fr.chi2    == approx(1 , rel=0.5)

def test_fits_yield_good_pulls():
    Nevt  = int(1e3)
    sigmas = np.random.uniform(low=1.0, high=50., size=30)
    means  = np.random.uniform(low=100, high=1000., size=30)

    SEED = []
    MU = []
    STD = []
    AVG = []
    RMS = []
    CHI2 = []
    n_sigma =3
    for sigma in sigmas:
        for mean in means:
            SEED.append(Measurement(mean, sigma))
            e  = np.random.normal(mean, sigma, Nevt)
            r = mean - n_sigma * sigma, mean + n_sigma * sigma
            bin_size = (r[1] - r[0]) / 50

            gp = gaussian_parameters(e, range = r, bin_size=bin_size)
            fc = energy_fit(e, nbins=50, range=r, n_sigma = n_sigma)

            MU.append(Measurement(fc.fr.par[1], fc.fr.err[1]))
            STD.append(Measurement(fc.fr.par[2], fc.fr.err[2] ))
            AVG.append(gp.mu)
            RMS.append(gp.std)
            CHI2.append(fc.fr.chi2)

    mean = np.array([x.value for x in SEED])
    sigma = np.array([x.uncertainty for x in SEED])
    avg = np.array([x.value for x in AVG])
    avg_u = np.array([x.uncertainty for x in AVG])
    rms = np.array([x.value for x in RMS])
    rms_u = np.array([x.uncertainty for x in RMS])
    mu = np.array([x.value for x in MU])
    mu_u = np.array([x.uncertainty for x in MU])
    std = np.array([x.value for x in STD])
    std_u = np.array([x.uncertainty for x in STD])

    p_mu, p_std = mean_and_std((mean-mu) / mu_u, range_ =(-10,10))
    print(f'(mean-mu) / mu_u -> {p_mu}, {p_std}')
    assert p_mu   == approx(0,  abs=0.2)
    assert p_std  == approx(1,  abs=0.3)

    p_mu, p_std = mean_and_std((mean-avg) / avg_u, range_ =(-10,10))
    print(f'(mean-avg) / avg_u -> {p_mu}, {p_std}')
    assert p_mu   == approx(0,  abs=0.2)
    assert p_std  == approx(1,  abs=0.3)

    p_mu, p_std = mean_and_std((sigma-std) / std_u, range_ =(-10,10))
    print(f'(sigma-std) / std_u -> {p_mu}, {p_std}')
    assert p_mu   <  1.5
    assert p_std  == approx(1,  abs=0.3)


    p_mu, p_std = mean_and_std((sigma-rms) / rms_u, range_ =(-10,10))
    print(f'(sigma-rms) / rms_u -> {p_mu}, {p_std}')
    assert p_mu   < 1
    assert p_std  == approx(1,  abs=0.3)
