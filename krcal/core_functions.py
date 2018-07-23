import numpy as np
import tables as tb
import pandas as pd
import glob

from typing import TypeVar

Number = TypeVar('Number', int, float)

from   invisible_cities.core.core_functions import loc_elem_1d


def meam_and_std(x : np.array, range : Tuple[Number])->Tuple[Number]:
    """Computes mean and std for an array within a range"""
    x1 = loc_elem_1d(x, find_nearest(x,range[0]))
    x2 = loc_elem_1d(x, find_nearest(x,range[1]))
    xmin = min(x1, x2)
    xmax = max(x1, x2)

    mu, std = weighted_mean_and_std(x[xmin:xmax], np.ones(len(x[xmin:xmax])))
    return mu, std


def find_nearest(array : np.array, value : Number)->Number:
    """Return the array element nearest to value"""
    idx = (np.abs(array-value)).argmin()
    return array[idx]


def bin_ratio(array : np.array, bins : np.array, xbin : int)-> Number:
    """Return the ratio between the element array[xbin] and the array sum"""
    return array[loc_elem_1d(bins, xbin)] / np.sum(array)


def bin_to_last_ratio(array : np.array, bins : np.array, xbin : int)-> Number:
    return np.sum(array[loc_elem_1d(bins, xbin): -1]) / np.sum(array)


def divide_np_arrays(num : np.array, denom : np.array) -> np.array:
    """Safe division of two arrays"""
    assert len(num) == len(denom)
    ok    = denom > 0
    ratio = np.zeros(len(denom))
    np.divide(num, denom, out=ratio, where=ok)
    return ratio
