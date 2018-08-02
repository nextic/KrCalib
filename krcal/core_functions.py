import numpy as np
import tables as tb
import pandas as pd

import matplotlib.pyplot as plt

from typing import List, Tuple
from . kr_types import Number


from invisible_cities.core.core_functions import in_range

def mean_and_std(x : np.array, range_ : Tuple[Number])->Tuple[Number]:
    """Computes mean and std for an array within a range"""

    mu = np.mean(x[in_range(x, *range_)])
    std = np.std(x[in_range(x, *range_)])
    return mu, std


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
