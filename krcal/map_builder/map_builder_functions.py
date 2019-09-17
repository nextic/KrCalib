from typing import Tuple
from typing import Optional
import pandas as pd
import numpy  as np
from krcal.core.kr_types  import ASectorMap
from dataclasses import dataclass

@dataclass
class reference_histograms:
    Z_distribution_hist : np.array


def quality_cut(dst : pd.DataFrame, r_range : float, **kwargs) -> pd.DataFrame:
    """ Does basic quality cut : R inside the r_range"""
    pass

def load_data(config :str) -> Tuple(pd.DataFrame, ASectorMap, reference_histograms):
    """ Reads kdst files and applies basic R cut. Outputs kdst as pd.DataFrame,
    bootstrap map, and reference histograms"""
    pass


def check_ns_cut(dst         : pd.DataFrame,
                 s_type      : str,
                 threshold   : float,
                 mask        : Optional[np.array] = None,
                 output_hist : bool = True,
                 hist_folder : Optional[str] = None,
                 **kwargs                                  ) -> np.array:
    """ Checks percentage of events with nSi ==1, optionally saves histogram
    and outputs mask. Raises exception if number of events smaller than
    threshold."""
    pass

def check_z_dist(dst           : pd.DataFrame,
                 allowed_sigma : float,
                 mask          : Optional[np.array] = None,
                 **kwargs                                  ) -> None:
    """ checks the z distribution of events, raises exception ifç
    distribution differes"""
    pass

def rate_check(dst         : pd.DataFrame,
               sigma_range : float,
               mask        : Optional[np.array] = None,
               **kwargs                                ) -> None:
    """ Raises exception if rate changes more than sigma_range"""
    pass

def z_band_sel(dst           : pd.DataFrame,
               bootstrap_map : ASectorMap,
               sigma_range   : float,
               mask          : Optional[np.array] = None,
               **kwargs                                  ) -> np.array:
    """ Applies geometric corrections from the bootstrap map and outputs
    mask - events inside the band"""
    pass

def calculate_bins(dst       : pd.dataFrame,
                   threshold : float,
                   **kwargs                 ) -> Tuple(int, int):
    """ Calculates number of bins based on the number of events > threshold.
     Returns Tuple of bins"""
    pass



def calculate_map(dst : pd.DataFrame, bins : Tuple(int, int), **kwargs):
    """ Calculates and outputs correction map"""
    pass

def check_failed_fits(**kwargs):
    """ Raises exception if fit failed"""
    pass
def regularize_map(maps : ASectorMap, **kwargs) -> ASectorMap:
    """ Applies regularization to the map"""
    pass

def add_krevol(maps : ASectorMap, dst : pd.DataFrame, **kwargs) -> ASectorMap:
    """ Adds time evolution dataframe to the map"""
    pass

def write_map (maps : ASectorMap, map_folder : str, **kwargs ):
    """A function to write maps"""
    pass

def compute_map(dst : pd.DataFrame, bins : Tuple(int, int), **kwargs) -> ASectorMap:
    maps = calculate_map (dst, bins, **kwargs)
    check_failed_fits (**kwargs)
    regularized_maps = regularize_map(maps, **kwargs)
    add_krevol(regularized_maps, dst)
    return regularized_maps

def apply_cuts(dst : pd.DataFrame, allowed_z_sigma : float, **kwargs) -> pd.DataFrame:
    mask1 = check_ns_cut(dst, 'S1', **kwargs)
    mask2 = check_ns_cut(dst, 'S2', mask=mask1, **kwargs)
    check_z_dist(dst, **kwargs)
    mask3 = z_band_sel(dst, allowed_z_sigma, **kwargs)
    all_mask = mask1 * mask2 * mask3
    return dst[all_mask]

def automatic_test(config):
    dst, bootstrapmap, references  = load_data(**locals)
    rate_check (dst, **locals)
    dst_passed_cut = apply_cuts(dst, **locals)
    rate_check(dst_passed_cut, **locals)

    bin_size  = calculate_bins(dst_passed_cut, **locals)
    final_map = compute_map(dst_passed_cut, bin_size, **locals)

    write_map(final_map, **locals)
