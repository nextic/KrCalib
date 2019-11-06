"""
Tests for fit_functions
"""

import numpy                                as np

from   pytest                               import approx

from   hypothesis                           import given
from   hypothesis.strategies                import floats

import invisible_cities.database.load_db    as     DB
from invisible_cities.core                  import  fit_functions as fitf
from . fit_functions                        import sigmoid
from . fit_functions                        import compute_drift_v
from . fit_functions                        import chi2f


def test_get_chi2f_when_data_equals_error_and_fit_equals_zero():
    Nevt  = int(1e6)
    xdata = np.zeros(Nevt) # Dummy value, not needed
    ydata = np.random.uniform(1, 100, Nevt)
    errs  = ydata
    f     = lambda x: np.zeros_like(x)
    chi2 = chi2f(f, 0, xdata, ydata, errs)
    assert chi2   == approx(1  , rel=1e-3)



def test_get_chi2f_when_data_equals_fit():
    Nevt  = int(1e6)
    xdata = np.zeros(Nevt) # Dummy value, not needed
    ydata = np.random.uniform(1, 100, Nevt)
    f     = lambda x: ydata
    errs  = ydata**0.5 # Dummy value, not needed
    chi2  = chi2f(f, 0, ydata, ydata, errs)

    assert chi2   == approx(0., rel=1e-3)


def test_chi2f_str_line():
    def line(x, m, n):
        return m * x + n
    y  = np.array([ 9.108e3, 10.34e3,   1.52387e5,   1.6202e5])
    ey = np.array([ 3.17   , 13.5   ,  70        ,  21       ])
    x  = np.array([29.7    , 33.8   , 481        , 511       ])

    fit = fitf.fit(line, x, y, seed=(1,1), sigma=ey)
    f = lambda x: line(x, *fit.values)
    assert chi2f(f, 2, x, y, ey) == approx(14, rel=1e-02)


@given(floats(min_value = -1e4,
              max_value = +1e4),
       floats(min_value = -1e4,
              max_value = + 1e4))
def test_sigmoid_limits(A, D):
    aux_sigmoid = lambda x: sigmoid(x, A, 0, 10, D)
    assert aux_sigmoid(-10) == approx(D  , rel=1e-2)
    assert aux_sigmoid( 10) == approx(A+D, rel=1e-2)

@given(floats(min_value = -1e4,
              max_value = +1e4),
       floats(min_value = -1e4,
              max_value = + 1e4),
       floats(min_value = -1e2,
              max_value = + 1e2),
       floats(min_value = -1e4,
              max_value = + 1e4))
def test_sigmoid_values_at_abscissa_axis(A, B, C, D):
    value = sigmoid(0, A, B, C, D)
    test  = A / (1 + np.exp(B*C)) + D
    assert value == test


@flaky(max_runs=10, min_passes=9)
def test_compute_drift_v_when_moving_edge():
    edge    = np.random.uniform(530, 570)
    Nevents = 100 * 1000
    data    = np.random.uniform(450, edge, Nevents)
    data    = np.random.normal(data, 1)
    dv, dvu = compute_drift_v(data, 60, [500,600],
                              [1500, 550,1,0], 'new')
    dv_th   = DB.DetectorGeo('new').ZMAX[0]/edge

    assert dv_th == approx(dv, abs=5*dvu)
