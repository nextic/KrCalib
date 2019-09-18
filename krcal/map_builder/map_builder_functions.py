from typing import Tuple
from typing import Optional
import pandas as pd
import numpy  as np
from krcal.core.kr_types  import ASectorMap, type_of_signal
from dataclasses import dataclass
from krcal.core       .histo_functions    import compute_and_save_hist_as_pd
from krcal.core       .histo_functions    import compute_equal_histo
from krcal.core       .histo_functions    import normalize_histo_and_error
from krcal.core       .kr_parevol_functions import get_number_of_time_bins
from krcal.map_builder.checking_functions import check_if_values_in_interval

class AbortingMapCreation(Exception):
    pass

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


def selection_nS_mask_and_checking(dst        : pd.DataFrame                ,
                                   column     : type_of_signal              ,
                                   interval   : Tuple[float, float]         ,
                                   output_f   : pd.HDFStore                 ,
                                   nbins_hist : int                 = 10    ,
                                   range_hist : Tuple[float, float] = (0,10),
                                   norm       : bool = True)->np.array:
    """
    Selects nS1(or nS2) == 1 for a given kr dst and
    returns the mask. It also computes selection efficiency,
    checking if the value is within a given interval, and
    saves histogram parameters.
    Parameters
    ----------
    dst: pd.Dataframe
        Krypton dst dataframe.
    column: type_of_signal
        The function can be appplied over nS1 or nS2.
    interval: length-2 tuple
        If the selection efficiency is out of this interval
        (given by the config file) the map production will abort.
    output_f: pd.HDFStore
        File where histogram will be saved.
    nbins_hist: int
        Number of bins to make the histogram.
    range_hist: length-2 tuple (optional)
        Range of the histogram.
    norm: bool
        If True, histogram will be normalized.
    Returns
    ----------
        A mask corresponding to the selected events.
    """
    mask         = getattr(dst, column.value) == 1
    nevts_before = dst[mask].event.nunique()
    nevts_after  = dst      .event.nunique()
    eff          = nevts_before / nevts_after

    compute_and_save_hist_as_pd(getattr(dst[(dst.s1_peak==0)&(dst.s2_peak==0)],
                                        column.value),
                                output_f, column.value,
                                nbins_hist, range_hist, norm)

    message = "Selection efficiency of "
    message += column.value
    message += "==1 out of range."
    check_if_values_in_interval(np.array(eff),
                                interval[0],
                                interval[1],
                                message)
    return mask


def check_Z_dst(Z_vect   : np.array,
                ref_file : str     ,
                n_sigmas : int      = 10)->None:
    """
    From a given Z distribution, this function checks, raising
    an exception, if Z histogram is correct.
    Parameters:
    ----------
    Z_vect : np.array
        Array of Z values for each kr event.
    ref_file: string
        Name of the file that contains reference histogram.
        Table must contain two columns: 'Z' and 'entries'.
    n_sigmas: int
        Number of sigmas to consider if distributions are similar enough.
    Returns
    -------
        Continue if both Z distributions are compatible
        within a 'n_sigmas' interval.
    """
    ref = pd.read_hdf(ref_file, key='Z_hist')

    N_Z, z_Z   = compute_equal_histo(Z_vect, ref)
    N_Z, err_N = normalize_histo_and_error(N_Z, z_Z)

    diff     = N_Z - ref.entries
    diff_sig = diff / (err_N+ref.error)

    message = "Z distribution very different to reference one."
    message += " May be some error in Z distribution of events."
    check_if_values_in_interval(diff_sig, -n_sigmas, n_sigmas, message)
    return;


def check_rate_and_hist(times    : np.array           ,
                        output_f : pd.HDFStore        ,
                        n_dev    : float       = 5    ,
                        bin_size : int         = 180  ,
                        normed   : bool        = False)->None:
    """
    Raises exception if evolution of rate vs time is
    not flat. It also computes histogram.
    Parameters
    ----------
    times: np.array
        Time of the events.
    output_f: pd.HDFStore
        File where histogram will be saved.
    n_dev: float
        Relative standard deviation to judge if
        distribution is correct.
    bin_size: int
        Size (in seconds) for histogram bins.
        By default it corresponds to 3 min.
    normed: bool (optional)
        If True, histogram will be normalized.
    Returns
    ----------
        Nothing.

    """
    min_time   = times.min()
    max_time   = times.max()
    ntimebins  = get_number_of_time_bins(bin_size,
                                         min_time,
                                         max_time)
    compute_and_save_hist_as_pd(times,
                                output_f,
                                table_name = "Rate",
                                n_bins     = ntimebins,
                                range_hist = (min_time,
                                              max_time))

    n, _ = np.histogram(times, bins=ntimebins,
                        range = (min_time, max_time))
    mean    = np.mean(n)
    dev     = np.std(n, ddof = 1)
    rel_dev = dev / mean * 100

    message = "Relative deviation is greater than the allowed one."
    message += " There must be some issue in rate distribution."
    check_if_values_in_interval(np.array(rel_dev)        ,
                                low_lim         = 0      ,
                                up_lim          = n_dev  ,
                                raising_message = message)
    return;


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
