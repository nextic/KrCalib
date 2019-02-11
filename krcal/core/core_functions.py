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


def get_time_series_df(time_bins    : Number,
                       time_range   : Tuple[float, float],
                       dst          : DataFrame,
                       time_column  : str = 'DT')->Tuple[np.array, List[np.array]]:
    """

    Given a dst (DataFrame) with a time column specified by the name time,
    this function returns a time series (ts) and a list of masks which are used to divide
    the event in time tranches.

    More generically, one can produce a "time series" using any column of the dst
    simply specifying time_column = ColumName

        Parameters
        ----------
            time_bins
                Number of time bines.
            time_range
                Time range.
            dst
                A Data Frame 
            time_column
            A string specifyng the dst column to be divided in time slices.

        Returns
        -------
            A Tuple with:
            np.array       : This is the ts vector
            List[np.array] : This are the list of masks defining the events in the time series.

    """

    logging.debug(f'function: get_time_series')
    nt = time_bins
    x = int((time_range[-1] -  time_range[0]) / nt)
    tfirst = int(time_range[0])
    tlast  = int(time_range[-1])
    if x == 1:
        indx = [(tfirst, tlast)]
    else:
        indx = [(i, i + x) for i in range(tfirst, int(tlast - x), x) ]
        indx.append((x * (nt -1), tlast))

    ts = [(indx[i][0] + indx[i][1]) / 2 for i in range(len(indx))]

    logging.debug(f' number of time bins = {nt}, t_first = {tfirst} t_last = {tlast}')
    logging.debug(f'indx = {indx}')
    logging.debug(f'ts = {ts}')

    times = dst[time_column].values
    masks = [in_range(times, indx[i][0], indx[i][1]) for i in range(len(indx))]

    return np.array(ts), masks


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
