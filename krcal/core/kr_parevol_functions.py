from . fit_lt_functions     import fit_lifetime_profile
from . correction_functions import e0_xy_correction
from . fit_functions        import compute_drift_v
from . map_functions        import amap_max
from . kr_types             import ASectorMap

from   typing               import List, Tuple

import pandas as pd
import numpy  as np


def computing_kr_parameters(data       : pd.DataFrame,
                            ts         : float,
                            emaps      : ASectorMap,
                            xr_map     : Tuple[float, float],
                            yr_map     : Tuple[float, float],
                            nx_map     : int,
                            ny_map     : int,
                            zslices_lt : int,
                            zrange_lt  : Tuple[float,float]  = (0, 550),
                            nbins_dv   : int                 = 35,
                            zrange_dv  : Tuple[float, float] = (500, 625),
                            detector   : str                 = 'new',
                            plot_fit   : bool                = False)->pd.DataFrame:

    """
    Computes some average parameters (e0, lt, drift v,
    S1w, S1h, S1e, S2w, S2h, S2e, S2q, Nsipm, 'Xrms, Yrms)
    for a given krypton distribution. Returns a DataFrame.

    Parameters
    ----------
    data: DataFrame
        Kdst distribution to analyze.
    ts: float
        Central time of the distribution.
    emaps: correction map
        Allows geometrical correction of the energy.
    xr_map, yr_map: length-2 tuple
        Set the X/Y-coordinate range of the correction map.
    nx_map, ny_map: int
        Set the number of X/Y-coordinate bins for the correction map.
    zslices_lt: int
        Number of Z-coordinate bins for doing the exponential fit to compute the lifetime
    zrange_lt: length-2 tuple (optional)
        Number of Z-coordinate range for doing the exponential fit to compute the lifetime
    nbins_dv: int (optional)
        Number of bins in Z-coordinate for doing the histogram to compute the drift velocity
    zrange_dv: int (optional)
        Range in Z-coordinate for doing the histogram to compute the drift velocity
    detector: string (optional)
        Used to get the cathode position from DB for the drift velocity computation.
    plot_fit: boolean (optional)
        Flag for plotting the Z-distribution of events around the cathode and the sigmoid fit.

    Returns
    -------
    pars: DataFrame
        Each column corresponds to the average value of a different parameter.
    """

    ## lt and e0
    norm     = amap_max(emaps)
    _, _, fr = fit_lifetime_profile(data.Z,
                                    e0_xy_correction(data.S2e.values,
                                                     data.X.values,
                                                     data.Y.values,
                                                     E0M=(emaps.e0/norm.e0),
                                                     xr=xr_map, yr=yr_map,
                                                     nx=nx_map, ny=ny_map),
                                    zslices_lt, zrange_lt)
    e0,  lt  = fr.par
    e0u, ltu = fr.err

    ## compute drift_v
    dv, dvu  = compute_drift_v(data.Z, nbins=nbins_dv,
                               zrange=zrange_dv, detector=detector,
                               plot_fit=plot_fit)

    ## average values
    parameters = ['S1w', 'S1h', 'S1e', 'S2w', 'S2h', 'S2e', 'S2q', 'Nsipm', 'Xrms', 'Yrms']
    mean_dict, var_dict = {}, {}
    for parameter in parameters:
        data_value           = getattr(data, parameter)
        mean_dict[parameter] = np.mean(data_value)
        var_dict [parameter] = (np.var(data_value)/len(data_value))**0.5

    ## saving as pd.DataFrame
    pars = pd.DataFrame({'ts':    [ts],
                         'e0':    [e0],                 'e0u':    [e0u],
                         'lt':    [lt],                 'ltu':    [ltu],
                         'dv':    [dv],                 'dvu':    [dvu],
                         's1w':   [mean_dict['S1w']],   's1wu':   [var_dict['S1w']],
                         's1h':   [mean_dict['S1h']],   's1hu':   [var_dict['S1h']],
                         's1e':   [mean_dict['S1e']],   's1eu':   [var_dict['S1e']],
                         's2w':   [mean_dict['S2w']],   's2wu':   [var_dict['S2w']],
                         's2h':   [mean_dict['S2h']],   's2hu':   [var_dict['S2h']],
                         's2e':   [mean_dict['S2e']],   's2eu':   [var_dict['S2e']],
                         's2q':   [mean_dict['S2q']],   's2qu':   [var_dict['S2q']],
                         'Nsipm': [mean_dict['Nsipm']], 'Nsipmu': [var_dict['Nsipm']],
                         'Xrms':  [mean_dict['Xrms']],  'Xrmsu':  [var_dict['Xrms']],
                         'Yrms':  [mean_dict['Yrms']],  'Yrmsu':  [var_dict['Yrms']]})

    return pars


def kr_time_evolution(ts         : np.array,
                      masks      : List[np.array],
                      dst        : pd.DataFrame,
                      emaps      : ASectorMap,
                      xr_map     : Tuple[float, float],
                      yr_map     : Tuple[float, float],
                      nx_map     : int,
                      ny_map     : int,
                      zslices_lt : int                 = 50,
                      zrange_lt  : Tuple[float,float]  = (0, 550),
                      nbins_dv   : int                 = 35,
                      zrange_dv  : Tuple[float, float] = (500, 625),
                      detector   : str                 = 'new',
                      plot_fit   : bool                = False)->pd.DataFrame:
    """
    Computes some average parameters (e0, lt, drift v,
    S1w, S1h, S1e, S2w, S2h, S2e, S2q, Nsipm, 'Xrms, Yrms)
    for a given krypton distribution and for different time slices.
    Returns a DataFrame.

    Parameters
    ----------
    ts: np.array of floats
        Sequence of central times for the different time slices.
    masks: list of boolean lists
        Allows dividing the distribution into time slices
    data: DataFrame
        Kdst distribution to analyze.
    emaps: correction map
        Allows geometrical correction of the energy.
    xr_map, yr_map: length-2 tuple
        Set the X/Y-coordinate range of the correction map.
    nx_map, ny_map: int
        Set the number of X/Y-coordinate bins for the correction map.
    z_slices: int (optional)
        Number of Z-coordinate bins for doing the exponential fit to compute the lifetime
    zrange_lt: length-2 tuple (optional)
        Number of Z-coordinate range for doing the exponential fit to compute the lifetime
    nbins_dv: int (optional)
        Number of bins in Z-coordinate for doing the histogram to compute the drift velocity
    zrange_dv: int (optional)
        Range in Z-coordinate for doing the histogram to compute the drift velocity
    detector: string (optional)
        Used to get the cathode position from DB for the drift velocity computation.
    plot_fit: boolean (optional)
        Flag for plotting the Z-distribution of events around the cathode and the sigmoid fit.

    Returns
    -------
    pars: DataFrame
        Each column corresponds to the average value for a given parameter.
        Each row corresponds to the parameters for a given time slice.
    """

    frames = []
    for index in range(len(masks)):
        sel_dst = dst[masks[index]]
        pars    = computing_kr_parameters(sel_dst, ts[index],
                                          emaps, xr_map, yr_map, nx_map, ny_map,
                                          zslices_lt, zrange_lt,
                                          nbins_dv, zrange_dv,
                                          detector, plot_fit)
        frames.append(pars)

    total_pars = pd.concat(frames, ignore_index=True)

    return total_pars
