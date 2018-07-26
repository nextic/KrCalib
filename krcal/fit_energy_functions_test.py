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

from . fit_energy_functions       import gaussian_parameters
from . fit_energy_functions       import energy_fit
from . fit_functions              import chi2


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
