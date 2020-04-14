"""
Tests for fit_functions
"""

import numpy as np
from pytest                import approx
from hypothesis            import given
from hypothesis            import settings
from hypothesis.strategies import floats

from .. core. kr_types                  import Measurement
from .. core. stat_functions            import mean_and_std
from .. core. testing_utils             import gaussian_experiments
from .  fit_energy_functions            import gaussian_parameters
from .  fit_energy_functions            import fit_energy
from .  fit_energy_functions            import fit_gaussian_experiments
from .  fit_energy_functions            import gaussian_params_from_fcs

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
    fc  = fit_energy (e , bins, range, n_sigma = n_sigma)
    par  = fc.fr.par

    mu_fit  = par[1]
    std_fit = par[2]
    assert gp.mu.value   == approx(mu_fit,  abs=0.3)
    assert gp.std.value  == approx(std_fit,  abs=0.3)
    assert fc.fr.chi2    == approx(1 , rel=0.5)


def test_fits_yield_good_pulls_fixed_mean_and_std():
    mean = 1e+4
    std  = 0.02
    sigma = mean * std
    exps = gaussian_experiments(mexperiments = 1000, nsample =1000, mean=mean, std = sigma)
    fcs = fit_gaussian_experiments(exps, nbins = 50, range =(9e+3, 11e+3), n_sigma =3)
    mus, umus, stds, ustds, chi2s = gaussian_params_from_fcs(fcs)


    p_mu, p_std = mean_and_std((mus-mean) / umus, range_ =(-5,5))
    print(f'mean: mu, std: -> {p_mu}, {p_std}')
    assert p_mu   == approx(0,  abs=0.1)
    assert p_std   == approx(1,  abs=0.2)

    p_mu, p_std = mean_and_std((stds-sigma) / ustds, range_ =(-5,5))
    print(f'mean: mu, std: -> {p_mu}, {p_std}')
    assert p_mu   <1 # std is biased
    assert p_std   == approx(1,  abs=0.2)



def test_fits_yield_good_pulls_variable_mean_and_std():
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
            fc = fit_energy(e, nbins=50, range=r, n_sigma = n_sigma)

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
