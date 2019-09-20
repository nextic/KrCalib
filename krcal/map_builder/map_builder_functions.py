from typing      import Tuple
from dataclasses import dataclass
import pandas as pd
import numpy  as np
import glob

from krcal.core.kr_types  import ASectorMap, type_of_signal
from krcal.core.selection_functions           import selection_in_band

from krcal.core       .io_functions       import write_complete_maps
from krcal.core       .histo_functions    import compute_and_save_hist_as_pd
from krcal.core       .histo_functions    import compute_similar_histo
from krcal.core       .histo_functions    import normalize_histo_and_poisson_error
from krcal.core       .kr_parevol_functions import get_number_of_time_bins
from krcal.map_builder.checking_functions import check_if_values_in_interval

from invisible_cities.core.core_functions  import in_range
from invisible_cities.io  .dst_io          import load_dsts
from invisible_cities.reco.corrections_new import read_maps
from invisible_cities.reco.corrections_new import maps_coefficient_getter
from invisible_cities.reco.corrections_new import correct_geometry_
from invisible_cities.reco.corrections_new import norm_strategy
from invisible_cities.reco.corrections_new import get_normalization_factor



def e0_xy_correction(map        : ASectorMap                         ,
                     norm_strat : norm_strategy   = norm_strategy.max):
    """
    Temporal function to perfrom IC geometric corrections only
    """
    normalization   = get_normalization_factor(map        , norm_strat)
    get_xy_corr_fun = maps_coefficient_getter (map.mapinfo, map.e0)
    def geo_correction_factor(x : np.array,
                              y : np.array) -> np.array:
        return correct_geometry_(get_xy_corr_fun(x,y))* normalization
    return geo_correction_factor

@dataclass
class reference_histograms:
    Z_distribution_hist : np.array


def quality_cut(dst : pd.DataFrame, r_max : float) -> pd.DataFrame:
    """
    Does basic quality cut : R inside the r_max
    Parameters
    ----------
    dst : pd.DataFrame
        Input dst to obtain the mask from.
    r_max: float
        Upper limit for R.
    Returns
    ----------
    mask : pd.DataFrame
        Mask for filtering events not matching the criteria
    """
    mask = in_range(dst.R, 0, r_max)
    return mask

def load_data(input_path         : str ,
              input_dsts         : str ,
              file_bootstrap_map : str ,
              ref_Z_histo_file   : str ,
              quality_ranges     : dict ) -> Tuple[pd.DataFrame,
                                                   ASectorMap  ,
                                                   reference_histograms]:
    """
    Reads kdst files and applies basic R cut. Outputs kdst as pd.DataFrame,
    bootstrap map, and reference histograms

    Parameters
    ----------
    input_path : str
        Path to the input map_folder
    input_dsts : str
        Name criteria for the dst to be read
    file_bootstrap_map : str
        Path to the bootstrap map file
    ref_Z_histo_file : str
        Path to the reference histogram file
    quality_ranges : dict
        Dictionary containing ranges for the quality cuts

    Returns
    ----------
    dst_filtered : pd.DataFrame
        Dst containing all the events once filtered
    bootstrap_map : ASectorMap
        Bootstrap map
    reference_histograms : reference_histograms
        To be completed
    """

    dst_files = glob.glob(input_path + input_dsts)
    dst_full  = load_dsts(dst_files, "DST", "Events")
    mask_quality = quality_cut(dst_full, **quality_ranges)
    dst_filtered = dst_full[mask_quality]

    bootstrap_map = read_maps(file_bootstrap_map)

    temporal = reference_histograms(None)

    return dst_filtered, bootstrap_map, temporal


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

    compute_and_save_hist_as_pd(values     = getattr(dst[(dst.s1_peak==0)&(dst.s2_peak==0)],
                                                     column.value),
                                out_file   = output_f,
                                hist_name  = column.value,
                                n_bins     = nbins_hist,
                                range_hist = range_hist,
                                norm       = norm)

    message = "Selection efficiency of "
    message += column.value
    message += "==1 out of range."
    check_if_values_in_interval(values          = np.array(eff),
                                low_lim         = interval[0]  ,
                                up_lim          = interval[1]  ,
                                raising_message = message      )
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

    N_Z, z_Z   = compute_similar_histo(param     = Z_vect,
                                       reference = ref   )
    N_Z, err_N = normalize_histo_and_poisson_error(N = N_Z,
                                                   b = z_Z)

    diff     = N_Z - ref.entries
    diff_sig = diff / (err_N+ref.error)

    message = "Z distribution very different to reference one."
    message += " May be some error in Z distribution of events."
    check_if_values_in_interval(values          = diff_sig ,
                                low_lim         = -n_sigmas,
                                up_lim          = n_sigmas ,
                                raising_message = message  )
    return;


def check_rate_and_hist(times      : np.array           ,
                        output_f   : pd.HDFStore        ,
                        name_table : str                ,
                        n_dev      : float       = 5    ,
                        bin_size   : int         = 180  ,
                        normed     : bool        = False)->None:
    """
    Raises exception if evolution of rate vs time is
    not flat. It also computes histogram.
    Parameters
    ----------
    times: np.array
        Time of the events.
    output_f: pd.HDFStore
        File where histogram will be saved.
    name_table: string
        Name for the histogram table inside file.
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

    compute_and_save_hist_as_pd(values    = times      ,
                                out_file  = output_f   ,
                                hist_name = name_table ,
                                n_bins    = ntimebins  ,
                                range_hist= (min_time  ,
                                             max_time) ,
                                norm      = normed     )

    n, _ = np.histogram(times, bins=ntimebins,
                        range = (min_time, max_time))
    mean    = np.mean(n)
    dev     = np.std(n, ddof = 1)
    rel_dev = dev / mean * 100

    message = "Relative deviation is greater than the allowed one."
    message += " There must be some issue in rate distribution."
    check_if_values_in_interval(values          = np.array(rel_dev),
                                low_lim         = 0                ,
                                up_lim          = n_dev            ,
                                raising_message = message          )
    return;


def band_selector_and_check(dst       : pd.DataFrame,
                           boot_map   : ASectorMap,
                           norm_strat : norm_strategy             = norm_strategy.max,
                           range_Z    : Tuple[np.array, np.array] = (10, 550),
                           range_E    : Tuple[np.array, np.array] = (10.0e+3,14e+3),
                           nbins_z    : int                       = 50,
                           nbins_e    : int                       = 50,
                           nsigma_sel : float                     = 3.5,
                           eff_min   : float                      = 0.4,
                           eff_max   : float                      = 0.6
                           )->np.array:
    """
    This function returns a selection of the events that
    are inside the Kr E vz Z band, and checks
    if the selection efficiency is correct.

    Parameters
    ----------
    dst : pd.DataFrame
        Krypton dataframe.
    boot_map: str
        Name of bootstrap map file.
    norm_strt: norm_strategy
        Provides the desired normalization to be used.
    range_Z: Tuple[np.array, np.array]
        Range in Z-axis
    range_E: Tuple[np.array, np.array]
        Range in Energy-axis
    nbins_z: int
        Number of bins in Z-axis
    nbins_e: int
        Number of bins in energy-axis
    nsigma_sel: float
        Number of sigmas to set the band width
    eff_min: float
        Lower limit of the range where selection efficiency
        is considered correct.
    eff_max: float
        Upper limit of the range where selection efficiency
        is considered correct.
    Returns
    ----------
        A  mask corresponding to the selection made.
    """
    emaps = e0_xy_correction(boot_map, norm_strat  = norm_strat)
    E0 = dst.S2e.values * emaps(dst.X.values, dst.Y.values)

    sel_krband, _, _, _, _ = selection_in_band(dst.Z, E0,
                                               range_z = range_Z,
                                               range_e = range_E,
                                               nbins_z = nbins_z,
                                               nbins_e = nbins_e,
                                               nsigma  = nsigma_sel)

    effsel = dst[sel_krband].event.nunique()/dst.event.nunique()
    message = "Band selection efficiency out of range."
    check_if_values_in_interval(values          = np.array(effsel),
                                low_lim         = eff_min         ,
                                up_lim          = eff_max         ,
                                raising_message = message         )

    return sel_krband

def get_binning_auto(nevt_sel: int,
                     thr_events_for_map_bins: int = 1e6,
                     n_bins:                  int = None)->int:
    """
    Computes the number of X-Y bins to be used in the creation
    of correction map regarding the number of selected events.
    Parameters
    ---------
    nevt_sel: int
        Number of kr events to compute the map.
    thr_events_for_map_bins: int (optional)
        Threshold to use 50x50 or 100x100 maps (standard values).
    n_bins: int (optional)
        The number of events to use can be chosen a priori.
    Returns
    ---------
    n_bins: int
        Number of bins in each direction (X,Y) (square map).
    """
    if n_bins != None: pass;
    elif nevt_sel<thr_events_for_map_bins:
        n_bins = 50;
    else: n_bins = 100
    return n_bins;



def calculate_map(dst : pd.DataFrame, nbins : int, **kwargs):

    dst_time = dst.sort_values('time')
    T       = dst_time.time.values
    DT      = time_delta_from_time(T)

    RMAX      = 200
    RCORE     = 100
    s1e_range = (3, 25)
    s2e_range = (2000, 14000)
    s2q_range = (200, 800)

    xy_range  = (-RMAX,  RMAX)
    z_range   = (10,  550)
    e_range = (5000, 14000)
    lt_range = (1000, 8000)
    c2_range = (0,5)

    krTimes, krRanges, krNbins, krBins = kr_ranges_and_bins(dst,
                                                            xxrange   = xy_range,
                                                            yrange    = xy_range,
                                                            zrange    = z_range,
                                                            s2erange  = s2e_range,
                                                            s1erange  = s1e_range,
                                                            s2qrange  = s2q_range,
                                                            xnbins    = nbins,
                                                            ynbins    = nbins,
                                                            znbins    = 15,
                                                            s2enbins  = 25,
                                                            s1enbins  = 10,
                                                            s2qnbins  = 25,
                                                            tpsamples = 3600) # tsamples in seconds

    KRES = select_xy_sectors_df(dst, krBins.X, krBins.Y)

    neM = event_map_df(KRES)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fpmxy = fit_map_xy_df(selection_map = KRES,
                           event_map     = neM,
                           n_time_bins   = 1,
                           time_diffs    = DT,
                           nbins_z        = krNbins.Z,
                           nbins_e        = krNbins.S2e,
                           range_z        = z_range,
                           range_e        = e_range,
                           energy        = 'S2e',
                           fit           = FitType.unbined,
                           n_min         = 100)

    tsm = tsmap_from_fmap(fpmxy)

    am = amap_from_tsmap(tsm,
                         ts = 0,
                         range_e     = e_range,
                         range_chi2  = c2_range,
                         range_lt    = lt_range)

    return am

def check_failed_fits(maps : ASectorMap, maxFailed : float = 600 ):

    numFailed = 0

    for i in range(len(maps.lt)):
        for j in range(len(maps.lt[i])):
            if np.isnan(maps.lt[i][j]):
                numFailed += 1

    if numFailed > maxFailed:
        raise NameError('NUM FILLED FITS EXCEEDS MAX ALLOWED')
    else:
        pass

def regularize_map(maps : ASectorMap, x2range : Tuple[float, float] = (0, 2) ):

    amap = asm_copy(maps)

    for i in range(len(amap.lt)):
        for j in range(len(amap.lt[i])):
            if amap.chi2[i][j] > x2range[1] or amap.chi2[i][j] < x2range[0]:
                amap.lt[i][j] = np.nan
                amap.ltu[i][j] = np.nan
                amap.e0[i][j] = np.nan
                amap.e0u[i][j] = np.nan

    av = amap_average(am)
    amap = amap_replace_nan_by_mean(amap, amMean=av)

    return amap

def add_krevol(maps : ASectorMap, dst : pd.DataFrame, **kwargs) -> ASectorMap:
    """ Adds time evolution dataframe to the map"""
    pass


def compute_map(dst : pd.DataFrame, bins : Tuple[int, int], **kwargs) -> ASectorMap:
    maps = calculate_map (dst, bins, **kwargs)
    check_failed_fits (**kwargs)
    regularized_maps = regularize_map(maps, **kwargs)
    add_krevol(regularized_maps, dst)
    return regularized_maps

def apply_cuts(dst              : pd.DataFrame       ,
               S1_signal        : type_of_signal     ,
               nS1_eff_interval : Tuple[float, float],
               store_hist_s1    : pd.HDFStore        ,
               ns1_histo_params : dict               ,
               S2_signal        : type_of_signal     ,
               nS2_eff_interval : Tuple[float, float],
               store_hist_s2    : pd.HDFStore        ,
               ns2_histo_params : dict               ,
               ref_Z_histo      : str                ,
               nsigmas_Zdst     : float              ,
               bootstrapmap     : ASectorMap         ,
               band_sel_params  : dict               ,
               ) -> pd.DataFrame:

    mask1 = selection_nS_mask_and_checking(dst = dst                  ,
                                           column = S1_signal         ,
                                           interval = nS1_eff_interval,
                                           output_f = store_hist_s1   ,
                                           **ns1_histo_params         )
    mask2 = selection_nS_mask_and_checking(dst = dst[mask1]           ,
                                           column = S2_signal         ,
                                           interval = nS2_eff_interval,
                                           output_f = store_hist_s2   ,
                                           **ns2_histo_params         )
    check_Z_dst(Z_vect   = dst.Z       ,
                ref_file = ref_Z_histo ,
                n_sigmas = nsigmas_Zdst)
    mask3 = band_selector_and_check(dst      = dst[mask1*mask2],
                                    boot_map = bootstrapmap    ,
                                    **band_sel_params          )
    all_mask = mask1 * mask2 * mask3
    return dst[all_mask]

def automatic_test(config):
    dst, bootstrapmap, references  = load_data(**locals)
    store_hist = pd.HDFStore(config.file_out_hists, "w", complib=str("zlib"), complevel=4)

    check_rate_and_hist(times      = dst.time         ,
                        output_f   = store_hist       ,
                        name_table = "rate_before_sel",
                        n_dev      = config.n_dev_rate,
                        **config.rate_histo_params    )

    dst_passed_cut = apply_cuts(dst              = dst                    ,
                                S1_signal        = type_of_signal.nS1     ,
                                nS1_eff_interval = (config.nS1_eff_min    ,
                                                    config.nS1_eff_max)   ,
                                store_hist_s1    = store_hist             ,
                                ns1_histo_params = config.ns1_histo_params,
                                S2_signal        = type_of_signal.nS2     ,
                                nS2_eff_interval = (config.nS2_eff_min    ,
                                                    config.nS2_eff_max)   ,
                                store_hist_s2    = store_hist             ,
                                ns2_histo_params = config.ns2_histo_params,
                                ref_Z_histo      = config.ref_Z_histo_file,
                                nsigmas_Zdst     = config.nsigmas_Zdst    ,
                                bootstrapmap     = bootstrapmap           ,
                                band_sel_params  = config.band_sel_params )

    check_rate_and_hist(times      = dst_passed_cut.time,
                        output_f   = store_hist         ,
                        name_table = "rate_after_sel"   ,
                        n_dev      = config.n_dev_rate  ,
                        **config.rate_histo_params      )

    store_hist.close()

    number_of_bins = get_binning_auto(nevt_sel                = dst_passed_cut.event.nunique()  ,
                                      thr_events_for_map_bins = config.thr_evts_for_sel_map_bins,
                                      n_bins                  = config.default_n_bins           )

    final_map = compute_map(dst_passed_cut, number_of_bins, **locals)

    write_complete_maps(asm      = final_map          ,
                        filename = config.file_out_map)
