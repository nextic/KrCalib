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

from .                     import fit_functions_ic as fitf

from . fit_functions       import chi2, expo_seed
from . stat_functions      import mean_and_std
from . stat_functions      import energy_lt_experiment
from . stat_functions      import energy_lt_experiments

from . fit_lt_functions    import fit_lifetime
from . fit_lt_functions    import fit_lifetime_profile
from . fit_lt_functions    import fit_lifetime_unbined
from . fit_lt_functions    import fit_lifetime_experiments
from . fit_lt_functions    import fit_lifetime_unbined
from . fit_lt_functions    import lt_params_from_fcs
from . kr_types import FitType

def test_lt_profile_yields_same_result_expo_fit():

    Nevt  = int(1e5)
    e0 = 1e+4 # pes
    std = 0.05 * e0
    lt = 2000 # lifetime in mus
    nbins_z = 12
    range_z = (1, 500)
    z, es = energy_lt_experiment(Nevt, e0, lt, std)

    x, y, yu     = fitf.profileX(z, es, nbins_z, range_z)
    valid_points = ~np.isnan(yu)

    x    = x [valid_points]
    y    = y [valid_points]
    yu   = yu[valid_points]
    seed = expo_seed(x, y)
    f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)

    c2   = chi2(f, x, y, yu)
    par  = np.array(f.values)
    err  = np.array(f.errors)
    e0   = par[0]
    lt   = - par[1]
    e0_u = err[0]
    lt_u = err[1]

    _, _,  fr = fit_lifetime_profile(z, es, nbins_z, range_z)
    assert e0   == approx(fr.par[0],  rel=0.05)
    assert lt   == approx(fr.par[1],  rel=0.05)
    assert e0_u == approx(fr.err[0],  rel=0.05)
    assert lt_u == approx(fr.err[1],  rel=0.05)
    #assert c2   == approx(fr.chi2,  rel=0.1)


@given(floats(min_value = 0.01,
              max_value = 0.1),
       floats(min_value = 100,
              max_value = 10000))
@settings(max_examples=10)
def test_lt_profile_yields_compatible_results_with_unbined_fit(sigma, lt):
    Nevt  = int(1e4)
    e0 = 1e+4 # pes
    std = sigma * e0
    lt = 2000 # lifetime in mus
    nbins_z = 12
    range_z = (1, 500)
    z, es = energy_lt_experiment(Nevt, e0, lt, std)

    _, _,  frp = fit_lifetime_profile(z, es, nbins_z, range_z)
    _, _,  fru = fit_lifetime_unbined(z, es, nbins_z, range_z)

    assert frp.par[0] == approx(fru.par[0],  rel=0.1)
    assert frp.par[1] == approx(fru.par[1],  rel=0.1)
    assert frp.err[0] == approx(fru.err[0],  rel=0.1)
    assert frp.err[1] == approx(fru.err[1],  rel=0.5)
    assert frp.chi2   == approx(fru.chi2,    rel=0.5)


def test_fit_lifetime_experiments_yield_good_pars_and_pulls():
    mexperiments = 1e+3
    nsample      = 1e+3
    e0 = 1e+4 # pes
    std = 0.05 * e0
    lt = 2000 # lifetime in mus

    zs, es = energy_lt_experiments(mexperiments, nsample, e0, lt, std)
    fcp = fit_lifetime_experiments(zs, es, nbins_z=12, nbins_e = 50,
                                   range_z = (1, 500), range_e = (7e+3, 11e+3),
                                   fit=FitType.profile)
    e0s, ue0s, lts,ults, chi2p = lt_params_from_fcs(fcp)

    p_e0, p_e0u = mean_and_std(e0s,   range_ =(e0 - 100, e0 + 100))
    p_lt, p_ltu = mean_and_std(lts,   range_ =(lt - 150, lt + 150))
    p_c2, p_c2u = mean_and_std(chi2p, range_ =(0, 2))
    assert p_e0   == approx(e0,  rel=0.01)
    assert p_lt   == approx(lt,  rel=0.01)
    assert p_c2   == approx(1,   rel=0.5)
    #assert p_c2u  == approx(0.2, rel=0.5)

    p_mu, p_std = mean_and_std((e0s-e0) / ue0s, range_ =(-5,5))
    assert p_mu   == approx(0,  abs=0.1)
    assert p_std  == approx(1,  rel=0.1)

    p_mu, p_std = mean_and_std((lts-lt) / ults, range_ =(-5,5))
    assert p_mu   == approx(0,  abs=0.1)
    assert p_std  == approx(1,  rel=0.1)
    p_c2, p_c2u = mean_and_std(chi2p, range_ =(0, 2))
    zs, es = energy_lt_experiments(mexperiments, nsample, e0, lt, std)
    fcp = fit_lifetime_experiments(zs, es, nbins_z=12, nbins_e = 50,
                                   range_z = (1, 500), range_e = (7e+3, 11e+3),
                                   fit=FitType.unbined)
    e0s, ue0s, lts,ults, chi2p = lt_params_from_fcs(fcp)

    p_e0, p_e0u = mean_and_std(e0s, range_ =(e0 - 100, e0 + 100))
    p_lt, p_ltu = mean_and_std(lts, range_ =(lt - 150, lt + 150))


    assert p_e0   == approx(e0,  rel=0.01)
    assert p_lt   == approx(lt,  rel=0.01)
    p_mu, p_std = mean_and_std((e0s-e0) / ue0s, range_ =(-5,5))
    assert p_mu   <= 0  # the pull is biased
    assert p_std  == approx(1,  rel=0.1)

    p_mu, p_std = mean_and_std((lts-lt) / ults, range_ =(-5,5))
    assert p_mu   <= 0  # the pull is biased
    assert p_std  == approx(1,  rel=0.1)
    assert p_c2   == approx(1,   rel=0.5)
    #assert p_c2u  == approx(0.2, rel=0.5)
