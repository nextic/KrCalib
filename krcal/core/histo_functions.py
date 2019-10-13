import numpy as np
from dataclasses import dataclass
from invisible_cities.core import fit_functions as fitf
from typing import Tuple

@dataclass
class ref_hist:
    bin_centres     : np.array
    bin_entries     : np.array
    err_bin_entries : np.array


def profile1d(z : np.array,
              e : np.array,
              nbins_z : int,
              range_z : np.array)->Tuple[float, float, float]:
    """Adds an extra layer to profileX, returning only valid points"""
    x, y, yu     = fitf.profileX(z, e, nbins_z, range_z)
    valid_points = ~np.isnan(yu)
    x    = x [valid_points]
    y    = y [valid_points]
    yu   = yu[valid_points]
    return x, y, yu



def compute_similar_histo(param     : np.array,
                          reference : ref_hist
                          )-> Tuple[np.array, np.array]:
    """
    This function computes a histogram with the same
    bin_size and number of bins as a given one.
    Parameters
    ----------
    param : np.array
        Array to be represented in the histogram.
    reference: pd.DataFrame
        Dataframe with the information of a reference histogram.
    Returns
    ----------
        Two arrays with the entries and the limits of each bin.
    """
    bin_size   = np.diff(reference.bin_centres)[0]
    min_Z_hist = reference.bin_centres.values[ 0] - bin_size/2
    max_Z_hist = reference.bin_centres.values[-1] + bin_size/2
    N, b = np.histogram(param, bins = len(reference.bin_centres),
                        range =(min_Z_hist, max_Z_hist));
    return N, b


def normalize_histo_and_poisson_error(N : np.array,
                                      b : np.array
                                      )->Tuple[np.array,
                                               np.array]:
    """
    Computes poissonian error for each bin. Normalizes the histogram
    with its area, applying this factor also to the error.
    Parameters
    ----------
    N: np.array
        Array with the entries inside each bin.
    b: np.array
        Array with limits of each bin.
    Returns
    ----------
        The input N array normalized, and its error multiplied
        by same normalization value.
    """
    err_N = np.sqrt(N)

    norm  = 1/sum(N)/((b[-1]-b[0])/(len(b)-1))
    N     = N*norm
    err_N = err_N*norm

    return N, err_N
