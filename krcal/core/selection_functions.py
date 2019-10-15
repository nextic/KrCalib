"""Module selection_functions.
This module includes the functions related with selection of events.

Notes
-----
    KrCalib code depends on the IC library.
    Public functions are documented using numpy style convention

Documentation
-------------
    Insert documentation https

Author: JJGC
Last revised: Feb, 2019

"""
import numpy as np
import pandas as pd
import datetime
from   pandas.core.frame import DataFrame

import matplotlib.pyplot as plt
from . histo_functions import labels
from . kr_types        import PlotLabels

from . core_functions  import  value_from_measurement
from . core_functions  import  uncertainty_from_measurement
from . core_functions  import  NN

from   invisible_cities.icaro. hst_functions import shift_to_bin_centers
from   invisible_cities.core.core_functions  import in_range
from   invisible_cities.icaro. hst_functions import shift_to_bin_centers
from . fit_lt_functions     import fit_lifetime, fit_lifetime_unbined
from . fit_functions        import fit_slices_1d_gauss


from . kr_types             import Number, Int, Range, Array
from . kr_types             import KrBins, KrNBins, KrRanges, KrTimes
from . kr_types             import KrEvent, FitType, FitParTS
from . kr_types             import HistoPar2, ProfilePar, FitPar, KrSector
from . core_functions       import phirad_to_deg
from . histo_functions      import h1, h1d, h2, h2d, plot_histo

from typing      import List, Tuple, Sequence, Iterable, Dict
import dataclasses as dc

import sys
import logging
log = logging.getLogger()


def event_map_df(dstMap : Dict[int, List[DataFrame]])->DataFrame:
    """
    Compute a numerical map from a DataFrame map

    Parameters
    ----------
    dstMAP:
        A DataFrame map, e.g, a Dict[int, List[DataFrame]]

    Returns
    -------
        A DataFrame in which each entry corresponds to the length of the
        DataFrames in the map.

    """
    DLEN = {}
    for i, ldst in dstMap.items():
        DLEN[i] =[len(dst) for dst in ldst]

    return pd.DataFrame.from_dict(DLEN)


def get_time_series_df(time_bins    : Number,
                       time_range   : Tuple[float, float],
                       dst          : DataFrame,
                       time_column  : str = 'time')->Tuple[np.array, List[np.array]]:
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
    nt = time_bins
    ip = np.linspace(time_range[0], time_range[-1], time_bins+1)
    masks = np.array([in_range(dst[time_column].values, ip[i], ip[i + 1]) for i in range(len(ip) -1)])
    return shift_to_bin_centers(ip), masks


def select_xy_sectors_df(dst        : DataFrame,
                         bins_x     : np.array,
                         bins_y     : np.array)-> Dict[int, List[DataFrame]]:
    """
    Return a DataFrameMap of selections organized by xy sector
    DataFrameMap = Dict[int, List[DataFrame]]

    Parameters
    ----------
        dst:
        The input data frame.
        bins_x:
        An array of bins along x.
        bins_y:
        An array of bins along y.

    Returns
    -------
        A DataFrameMap of selections
        where for each x (the key in the dict) one has a list
        (corresponding to y cells) of DataFrame (the events selected)

    """
    dstMap = {}
    nbins_x = len(bins_x) -1
    nbins_y = len(bins_y) -1
    for i in range(nbins_x):
        dstx = dst[in_range(dst.X, *bins_x[i: i+2])]
        dstMap[i] = [dstx[in_range(dstx.Y, *bins_y[j: j+2])] for j in range(nbins_y) ]

    return dstMap


def select_rphi_sectors_df(dst     : DataFrame,
                           RPS     : Dict[int, List[KrSector]])-> Dict[int, List[DataFrame]]:
    """
    Return a dict of KrEvent organized by rphi sector.

    Parameters
    ----------
        dst:
        The input data frame.

        RPS:
        RPHI selection, a map defining the RPHI wedges.

    Returns
    -------
        A DataFrameMap of selections
        where for each radial sector (the key in the dict) one has a list
        (corresponding to the PHI sectors) of DataFrame (the events selected)

    """

    def selection_mask_rphi_sectors(dst     : DataFrame,
                                       RPS     : Dict[int, List[KrSector]])->Dict[int, np.array]:
        """Returns a dict of selections arranged in a dict of rphi sectors"""
        logging.debug(f' --selection_mask_rphi_sectors:')
        MSK = {}
        for i, rps in RPS.items():
            sel_mask = [in_range(dst.R,
                                 s.rmin,
                                 s.rmax).values & in_range(phirad_to_deg(dst.Phi),
                                                           s.phimin,
                                                           s.phimax).values for s in rps]
            MSK[i] = sel_mask

        return MSK


    MSK = selection_mask_rphi_sectors(dst, RPS)
    RGES = {}
    for i, msk in MSK.items():
        logging.debug(f' defining kr_event for sector {i}')
        RGES[i] = [dst[m] for m in msk]
    return RGES


def selection_in_band(z         : np.array,
                      e         : np.array,
                      range_z   : Range,
                      range_e   : Range,
                      nbins_z   : int     = 50,
                      nbins_e   : int     = 100,
                      nsigma    : float   = 3.5) ->Tuple[np.array, FitPar, FitPar,
                                                         HistoPar2, ProfilePar]:
    """ This returns a selection of the events that are inside the Kr E vz Z
    returns: np.array(bool)
    """

    zbins  = np.linspace(*range_z, nbins_z + 1)
    zerror = np.diff(zbins) * 0.5
    ebins  = np.linspace(*range_e, nbins_e + 1)
    zc = shift_to_bin_centers(zbins)

    sel_e = in_range(e, *range_e)
    mean, sigma, chi2, ok = fit_slices_1d_gauss(z[sel_e], e[sel_e], zbins, ebins, min_entries=5e2)
    e_mean  = mean.value
    e_sigma = sigma.value
    # 1. Profile of mean values of e in bins of z
    #zc, e_mean, e_sigma = fitf.profileX(z, e, nbins_z, range_z, range_e)


    #2. Fit two exponentials to e_mmean +- ns_igma * e_sigma defining a band

    y         = e_mean +  nsigma * e_sigma
    fph, _, _    = fit_lifetime_unbined(zc, y, nbins_z, range_z)
    y         = e_mean - nsigma * e_sigma
    fpl, _, _ = fit_lifetime_unbined(zc, y, nbins_z, range_z)

    # 3. Select events in the range defined by the band

    sel_inband = in_range(e, fpl.f(z), fph.f(z))

    # return data
    hp = HistoPar2(var = z,
                   nbins = nbins_z,
                   range = range_z,
                   var2 = e,
                   nbins2 = nbins_e,
                   range2 = range_e)

    pp = ProfilePar(x = zc, xu = zerror, y = e_mean, yu = e_sigma)

    return sel_inband, fpl, fph, hp, pp


def plot_selection_in_band(fpl    : FitPar,
                           fph    : FitPar,
                           hp     : HistoPar2,
                           pp     : ProfilePar,
                           nsigma : float   = 3.5,
                           figsize=(10,6)):
    z       = hp.var
    e       = hp.var2
    range_z = hp.range
    range_e = hp.range2
    nbins_z = hp.nbins
    nbins_e = hp.nbins2

    zc     = pp.x
    emean  = pp.y
    zerror = pp.xu
    esigma = pp.yu

    zbins  = np.linspace(*range_z, nbins_z + 1)
    ebins  = np.linspace(*range_e, nbins_e + 1)


    fig = plt.figure(figsize=figsize)
    pltLabels =PlotLabels(x='Energy-like', y='Events', title='true')
    ax      = fig.add_subplot(1, 1, 1)

    nevt, *_  = plt.hist2d (z, e, (zbins, ebins))

    plt.errorbar(zc, emean, esigma, zerror,
                 "kp", label="Kr peak energy $\pm 1 \sigma$")

    plt.plot    (zbins,  fpl.f(zbins),  "m", lw=2, label="$\pm "+str(nsigma)+" \sigma$ region")
    plt.plot    (zbins,  fph.f(zbins),  "m", lw=2)
    plt.legend()
    labels(PlotLabels("Drift time (µs)", "S2 energy (pes)", "Energy vs drift"))


def selection_info(sel, comment=''):
    nsel   = np.sum(sel)
    effsel = 100.*nsel/(1.*len(sel))
    s = f"Total number of selected candidates {comment}: {nsel} ({effsel:.1f} %)"
    print(s)
    return s
