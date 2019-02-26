"""
Tests for fit_functions
"""

import numpy                                as np

from   pytest                               import approx

from   hypothesis                           import given
from   hypothesis.strategies                import floats

from   invisible_cities.evm  .ic_containers import Measurement
import invisible_cities.database.load_db    as     DB

from . fit_functions                        import sigmoid
from . fit_functions                        import compute_drift_v


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

@given(floats(min_value = 530,
              max_value = 570))
def test_compute_drift_v_when_moving_edge(edge):
    Nevents = 100*1000
    data    = np.random.uniform(450, edge, Nevents)
    data    = np.random.normal(data, 1)
    dv, dvu = compute_drift_v(data, 60, [500,600],
                              [1500, 550,1,0], 'new',
                              plot_fit=False)
    dv_th   = DB.DetectorGeo('new').ZMAX[0]/edge

    assert dv_th == approx(dv, abs=5*dvu)
