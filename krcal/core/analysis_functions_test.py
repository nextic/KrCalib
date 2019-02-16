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
from . core_functions      import data_frames_are_identical
from . analysis_functions  import nmap
from . analysis_functions  import select_xy_sectors_df
import pytest


# def x_and_y_ranges(data : DataFrame, xb : np.array, yb : np.array, nbx :int, nby : int):
#     r = True
#     for i in range(nbx):
#         dstx = data[in_range(data.X, *xb[i: i+2])]
#         r & in_range(dstx.X.values, xb[i: i+2][0], xb[i: i+2][1]).all()
#         for j in range(nby):
#             dsty = dstx[in_range(dstx.Y, *yb[j: j+2])]
#             r & in_range(dsty.Y.values, yb[j: j+2][0], yb[j: j+2][1]).all()
#     return r
#

# def test_x_and_y_ranges(dstData):
#     dst, xb, yb, nbx, nby, _, _, _, _, _, _ = dstData
#
#     assert x_and_y_ranges(dst, xb, yb, nbx, nby)
#
#
# def test_select_xy_sectors_df(dstData):
#     data, xb, yb, nbx, nby, _, _, _, _, _, _ = dstData
#
#     selDict = {}
#     for i in range(nbx):
#         dstx = data[in_range(data.X, *xb[i: i+2])]
#         selDict[i] = [dstx[in_range(dstx.Y, *yb[j: j+2])] for j in range(nby) ]
#
#     sel = nmap(selDict)
#     selMap = select_xy_sectors_df(data, xb, yb)
#     sel2 = nmap(selMap)
#     assert data_frames_are_identical(sel, sel2)
