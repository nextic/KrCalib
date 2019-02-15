"""
Tests for analysis_functions

See: https://github.com/nextic/KrCalibNB/blob/krypton/tutorials/TestsForMaps.ipynb
"""

import numpy as np
import pandas as pd
import datetime
from   pandas.core.frame import DataFrame

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

from   invisible_cities.core.core_functions    import in_range
from . core_functions       import data_frames_are_identical
from . selection_functions  import event_map_df
from . selection_functions  import get_time_series_df
from . fit_lt_functions     import get_time_series
from . selection_functions  import select_xy_sectors_df

from . analysis_functions   import kr_event
import pytest

def test_event_map_df(dstData):
    dst, xb, yb, nbx, nby, _, _, _, _, _, _ = dstData

    dstMap = {0:[dst]}
    em = event_map_df(dstMap)
    assert em[0][0] == len(dst)

def test_get_time_series_df(dstData):
    """
    See: https://github.com/nextic/KrCalibNB/blob/krypton/tutorials/TestsForMaps.ipynb
    """
    dst, _, _, _, _, nt, t0, tf, step, indx, ts = dstData

    ts2, masks = get_time_series_df(nt,(t0, tf), dst, time_column='time')

    assert np.allclose(ts,ts2)

    t0 = int(t0)
    tf = int(tf)
    if step == 1:
        indx2 = [(t0, tl)]
    else:
        indx2 = [(i, i + step) for i in range(t0, int(tf - step), step) ]
        indx2.append((step * (nt -1), tf))

    assert indx == indx2

    for i in range(len(masks)-1):
        assert np.count_nonzero(masks[i]) == 4

    #print(np.count_nonzero(masks[-1]))
    #assert np.count_nonzero(masks[-1] == 3)


def test_get_time_series_df_gives_same_result_time_series(dstData):
    dst, _, _, _, _, nt, t0, tf, step, indx, ts = dstData

    ts, masks = get_time_series_df(nt,(t0, tf), dst, time_column='time')

    kge = kr_event(dst, dst.time.values, dst.X, dst.Y)
    ts2, masks2 = get_time_series(nt, (t0, tf), kge)

    assert np.allclose(ts,ts2)
    assert np.array(masks).all() == np.array(masks2).all()


def x_and_y_ranges(data : DataFrame, xb : np.array, yb : np.array, nbx :int, nby : int):
    r = True
    for i in range(nbx):
        dstx = data[in_range(data.X, *xb[i: i+2])]
        r & in_range(dstx.X.values, xb[i: i+2][0], xb[i: i+2][1]).all()
        for j in range(nby):
            dsty = dstx[in_range(dstx.Y, *yb[j: j+2])]
            r & in_range(dsty.Y.values, yb[j: j+2][0], yb[j: j+2][1]).all()
    return r


def test_x_and_y_ranges(dstData):
    dst, xb, yb, nbx, nby, _, _, _, _, _, _ = dstData

    assert x_and_y_ranges(dst, xb, yb, nbx, nby)


def test_select_xy_sectors_df(dstData):
    data, xb, yb, nbx, nby, _, _, _, _, _, _ = dstData

    selDict = {}
    for i in range(nbx):
        dstx = data[in_range(data.X, *xb[i: i+2])]
        selDict[i] = [dstx[in_range(dstx.Y, *yb[j: j+2])] for j in range(nby) ]

    sel = event_map_df(selDict)
    selMap = select_xy_sectors_df(data, xb, yb)
    sel2 = event_map_df(selMap)
    assert data_frames_are_identical(sel, sel2)
