import time
from   datetime import datetime
import numpy as np
import pandas as pd
from   pandas.core.frame import DataFrame

from   typing      import Tuple, List, Iterable
from . kr_types    import Number
from   numpy      import pi
from   invisible_cities.evm.ic_containers  import Measurement

NN = np.nan

import sys
import logging
log = logging.getLogger()


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


def in_range(data, minval=-np.inf, maxval=np.inf):
    """
    Find values in range [minval, maxval).

    Parameters
    ---------
    data : np.ndarray
        Data set of arbitrary dimension.
    minval : int or float, optional
        Range minimum. Defaults to -inf.
    maxval : int or float, optional
        Range maximum. Defaults to +inf.

    Returns
    -------
    selection : np.ndarray
        Boolean array with the same dimension as the input. Contains True
        for those values of data in the input range and False for the others.
    """
    return (minval <= data) & (data < maxval)


def data_frames_are_identical(df1 : DataFrame, df2 : DataFrame)->bool:
    """
    Compare two data frames

    Parameters
    ----------
    df1, df2:
        Data Frames to be compared

    Returns
    -------
        A bool: True if all elements of the DFs are identical False otherwise
    """

    df = df1 == df2 # the resulting df is a df of bools.

    # first all() gives a bool per column, creating a Series,
    # seond all() gives a bool for the Series
    return df.eq(True).all().all()


def phirad_to_deg(r : float)-> float:
    return (r + pi) * 180 / pi


def value_from_measurement(mL : Iterable[Measurement]) -> np.array:
    return np.array([m.value for m in mL])


def uncertainty_from_measurement(mL : Iterable[Measurement]) -> np.array:
    return np.array([m.uncertainty for m in mL])


def time_delta_from_time(T):
    return np.array([t - T[0] for t in T])
    # dt = [(datetime.fromtimestamp(ts[i]) - datetime.fromtimestamp(ts[0])).total_seconds()
    #         for i in range (len(ts))]
    # return np.array(dt)


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


def file_numbers_from_file_range(file_range : Tuple[int, int])->List[str]:
    numbers = range(*file_range)
    N=[]
    for number in numbers:
        if number < 10:
            N.append(f"000{number}")
        elif 10 <= number < 100:
            N.append(f"00{number}")
        elif 100 <= number < 1000:
            N.append(f"0{number}")
        else:
            N.append(f"{number}")

    return N
