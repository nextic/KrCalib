import time
import numpy as np
import logging
from   typing      import Iterable
from   numpy       import pi
from   invisible_cities.evm.ic_containers  import Measurement
from . kr_types import Number

NN = np.nan
log = logging.getLogger(__name__)


def timeit(f):
    """
    Decorator for function timing.
    """
    def time_f(*args, **kwargs):
        t0 = time.time()
        output = f(*args, **kwargs)
        print("Time spent in {}: {} s".format(f.__name__,
                                              time.time() - t0))
        return output
    return time_f

def phirad_to_deg(r : float)-> float:
    return (r + pi) * 180 / pi

def value_from_measurement(mL : Iterable[Measurement]) -> np.array:
    return np.array([m.value for m in mL])

def uncertainty_from_measurement(mL : Iterable[Measurement]) -> np.array:
    return np.array([m.uncertainty for m in mL])

def time_delta_from_time(T):
    return np.array([t - T[0] for t in T])

def find_nearest(array : np.array, value : Number)->Number:
    """Return the array element nearest to value"""
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def divide_np_arrays(num : np.array, denom : np.array) -> np.array:
    """Safe division of two arrays"""
    assert len(num) == len(denom)
    ok    = denom > 0
    ratio = np.zeros(len(denom))
    np.divide(num, denom, out=ratio, where=ok)
    return ratio


def resolution(values, errors = None, E_from=41.5, E_to=2458):
    """
    Compute resolution at E_from and resolution at E_to
    with uncertainty propagation.
    """
    if errors is None:
        errors = np.zeros_like(values)

    amp  ,   mu,   sigma, *_ = values
    u_amp, u_mu, u_sigma, *_ = errors

    r   = 235. * sigma/mu
    u_r = r * (u_mu**2/mu**2 + u_sigma**2/sigma**2)**0.5

    scale = (E_from/E_to)**0.5
    return Measurement(r        , u_r        ), \
           Measurement(r * scale, u_r * scale)

