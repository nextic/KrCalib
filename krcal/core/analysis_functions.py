"""Module analysis_functions.
This module includes the functions related with selection of events.

Notes
-----
    KrCalib code depends on the IC library.
    Public functions are documented using numpy style convention

Documentation
-------------
    Insert documentation https
"""
import numpy as np
import pandas as pd
import datetime
from   pandas.core.frame import DataFrame

import matplotlib.pyplot as plt
from . histo_functions import labels
from . kr_types        import PlotLabels

import invisible_cities.core.fit_functions as fitf
from   invisible_cities.core .stat_functions import poisson_sigma
from   invisible_cities.icaro. hst_functions import shift_to_bin_centers
from   invisible_cities.core.core_functions  import in_range

from . fit_lt_functions     import fit_lifetime, fit_lifetime_unbined
from . fit_functions        import fit_slices_1d_gauss

from . kr_types             import Number, Int, Range, Array
from . kr_types             import KrBins, KrNBins, KrRanges, KrTimes
from . kr_types             import KrEvent
from . kr_types             import HistoPar2, ProfilePar, FitPar, KrSector
from . core_functions       import phirad_to_deg
from .histo_functions    import h1, h1d, h2, h2d, plot_histo

from typing      import List, Tuple, Sequence, Iterable, Dict
import dataclasses as dc

import sys
import logging
log = logging.getLogger()

def kr_event(dst      : DataFrame,
             DT       : Array      = [],
             E        : Array      = [],
             Q        : Array      = [],
             sel_mask : Array      = [])->KrEvent:
    """
    Defines a  KrEvent (a subset of a DST)
    Parameters
    ----------
    dst:
        A DataFrame dst.
    DT:
        A vector of time differences.
    E:
        A vector of energies (often but not neccessary corrected energy).
    Q:
        A vector of energies (often but not neccessary corrected SiPM Q).

    sel_mask:
        A selection mask.

    Returns
    -------
        KrEvent

    @dataclass
    class CPoint:
        X   : np.array
        Y   : np.array
        Z   : np.array


    @dataclass
    class Point(CPoint):
        R   : np.array
        Phi : np.array

    @dataclass
    class KrEvent(Point):
        S2e  : Array
        S1e  : Array
        S2q  : Array
        T    : Array  # time
        DT   : Array  # time difference in seconds
        E    : Array
        Q    : Array

    """

    if len(DT) == 0:
        DT = np.zeros(len(dst))
    else:
        assert len(DT) == len(dst)

    if len(E) == 0:
        E = np.zeros(len(dst))
    else:
        assert len(E) == len(dst)

    if len(Q) == 0:
        Q = np.zeros(len(dst))
    else:
        assert len(Q) == len(dst)

    if len(sel_mask) > 0:
        assert len(sel_mask) == len(dst)


        return KrEvent(X   = dst.X.values[sel_mask],
                       Y   = dst.Y.values[sel_mask],
                       Z   = dst.Z.values[sel_mask],
                       R   = dst.R.values[sel_mask],
                       Phi = dst.Phi.values[sel_mask],
                       T   = dst.time.values[sel_mask],
                       DT  = DT[sel_mask],
                       S2e = dst.S2e.values[sel_mask],
                       S1e = dst.S1e.values[sel_mask],
                       S2q = dst.S2q.values[sel_mask],
                       E   = E[sel_mask],
                       Q   = Q[sel_mask])
    else:
        return KrEvent(X   = dst.X.values,
                       Y   = dst.Y.values,
                       Z   = dst.Z.values,
                       R   = dst.R.values,
                       Phi = dst.Phi.values,
                       T   = dst.time.values,
                       DT  = DT,
                       S2e = dst.S2e.values,
                       S1e = dst.S1e.values,
                       S2q = dst.S2q.values,
                       E   = E,
                       Q   = Q)


def kr_event_selection(kh : KrEvent, sel_mask : Array = [])->KrEvent:

    if len(sel_mask) > 0:
        assert len(sel_mask) == len(kh.X)

        return KrEvent(X   = kh.X[sel_mask],
                       Y   = kh.Y[sel_mask],
                       Z   = kh.Z[sel_mask],
                       R   = kh.R[sel_mask],
                       Phi = kh.Phi[sel_mask],
                       T   = kh.T[sel_mask],
                       DT  = kh.DT[sel_mask],
                       S2e = kh.S2e[sel_mask],
                       S1e = kh.S1e[sel_mask],
                       S2q = kh.S2q[sel_mask],
                       E   = kh.E[sel_mask],
                       Q   = kh.Q[sel_mask])
    else:
        return KrEvent(X   = kh.X,
                       Y   = kh.Y,
                       Z   = kh.Z,
                       R   = kh.R,
                       Phi = kh.Phi,
                       T   = kh.T,
                       DT  = kh.DT,
                       S2e = kh.S2e,
                       S1e = kh.S1e,
                       S2q = kh.S2q,
                       E   = kh.E,
                       Q   = kh.Q)

def kr_bins(xxrange   : Range = (-220,  220),
            yrange    : Range = (-220,  220),
            zrange    : Range = (100,  550),
            s2erange  : Range = (3e3, 13e3),
            s1erange  : Range = (1, 25),
            s2qrange  : Range = (200, 800),
            xnbins    : Int   = 60,
            ynbins    : Int   = 60,
            znbins    : Int   = 12,
            s2enbins  : Int   = 50,
            s1enbins  : Int   = 10,
            s2qnbins  : Int   = 25)->KrBins:

    Xbins      = np.linspace(*xxrange, xnbins + 1)
    Ybins      = np.linspace(*yrange, ynbins + 1)
    Xcenters   = shift_to_bin_centers(Xbins)
    Ycenters   = shift_to_bin_centers(Ybins)
    Xpitch     = np.diff(Xbins)[0]
    Ypitch     = np.diff(Ybins)[0]

    return KrBins(S2e  = np.linspace(*s2erange,  s2enbins + 1),
                  S1e  = np.linspace(*s1erange,  s1enbins + 1),
                  S2q = np.linspace(*s2qrange,  s2qnbins + 1),
                  X    = Xbins,
                  Y    = Ybins,
                  Z    = np.linspace(*zrange, znbins + 1),
                  Xc   = Xcenters,
                  Yc   = Ycenters,
                  Xp   = Xpitch,
                  Yp   = Ypitch,
                  T    = None)


def kr_ranges_and_bins(dst       : DataFrame,
                       xxrange   : Range = (-220,  220),
                       yrange    : Range = (-220,  220),
                       zrange    : Range = (100,  550),
                       s2erange  : Range = (3e3, 13e3),
                       s1erange  : Range = (1, 25),
                       s2qrange  : Range = (200, 800),
                       xnbins    : Int   = 60,
                       ynbins    : Int   = 60,
                       znbins    : Int   = 12,
                       s2enbins  : Int   = 50,
                       s1enbins  : Int   = 10,
                       s2qnbins  : Int   = 25,
                       tpsamples : Int  = 3600)->Tuple[KrTimes, KrRanges, KrNBins, KrBins]:

    krNbins = KrNBins  (S2e  = s2enbins,
                        S1e  = s1enbins,
                        S2q = s2qnbins,
                        X    = xnbins,
                        Y    = ynbins,
                        Z    = znbins,
                        T    = None)

    krRanges = KrRanges(S2e  = s2erange,
                        S1e  = s1erange,
                        S2q  = s2qrange,
                        X    = xxrange,
                        Y    = yrange,
                        Z    = zrange,
                        T    = None)

    krBins = kr_bins(xxrange, yrange, zrange, s2erange, s1erange, s2qrange,
                     xnbins, ynbins, znbins, s2enbins, s1enbins, s2qnbins)


    dst_time = dst.sort_values('event')
    T       = dst_time.time.values

    tstart  = T[0]
    tfinal  = T[-1]
    Trange  = (tstart,tfinal)
    ntimebins  = int( np.floor( ( tfinal - tstart) / tpsamples) )
    Tnbins     = np.max([ntimebins, 1])
    Tbins      = np.linspace( tstart, tfinal, ntimebins+1)

    krRanges.T = Trange
    krNbins.T  = Tnbins
    krBins.T   = Tbins
    times      = [np.mean([Tbins[t],Tbins[t+1]]) for t in range(Tnbins)]

    TL         = [(Tbins[t],Tbins[t+1]) for t in range(Tnbins)]
    timeStamps = list(map(datetime.datetime.fromtimestamp, times))
    krTimes    = KrTimes(times = times, timeStamps = timeStamps, TL = TL)

    return krTimes, krRanges, krNbins, krBins


def kre_concat(KRL : List[KrEvent])->KrEvent:

    if len(KRL) == 1:
        return KRL[0]

    krg0 = dc.asdict(KRL[0])

    for i in range(1, len(KRL)):
        krg1 = dc.asdict(KRL[i])
        C = [np.concatenate((krg0[key], krg1[key])) for key in krg0.keys()]
        krg0 =  dc.asdict(KrEvent(*C))

    C = [krg0[key] for key in krg0.keys()]
    return KrEvent(*C)




def fiducial_volumes(dst     : DataFrame,
                     dt      : np.array,
                     E       : np.array,
                     Q       : np.array,
                     sectors : Iterable[KrSector])->List[KrEvent]:


    return [kr_event(dst, dt, E, Q,
            sel_mask = (in_range(dst.R,
                                s.rmin,
                                s.rmax).values) & in_range(phirad_to_deg(dst.Phi),
                                                           s.phimin,
                                                           s.phimax).values) for s in sectors]

def fid_eff(dst  : DataFrame,
            kdst : KrEvent)->float:
    n_dst      = len(dst)
    n_kdst     = len(kdst.S2e)
    return  n_kdst / n_dst


def select_rphi_sectors(dst     : DataFrame,
                        dt      : np.array,
                        E       : np.array,
                        Q       : np.array,
                        RPS     : Dict[int, List[KrSector]])-> Dict[int, List[KrEvent]]:
    """
    Return a dict of KrEvent organized by rphi sector.

    Parameters
    ----------
        dst:
        The input data frame.

        dt:
        An array of time differences needed to compute the time masks.

        E:
        An energy vector (can contain the corrected energy in the PMTs).

        Q:
        An energy vector (can contain the corrected energy in the SiPMs).

        RPS:
        RPHI selection, a map defining the RPHI wedges.

    Returns
    -------
        A map of selections defined as Dict[int, List[KrEvent]]
        where for each radial sector (the key in the dict) one has a list
        (corresponding to the PHI sectors) of KrEvent (the events selected)

    """

    def selection_mask_rphi_sectors(dst     : DataFrame,
                                    RPS     : Dict[int, List[KrSector]])->Dict[int, np.array]:
        """Returns a dict of selections arranged in a dict of rphi sectors"""
        logging.debug(f' --selection_mask_rphi_sectors:')
        MSK = {}
        for i, rps in RPS.items():
            logging.debug(f' computing mask in sector {i}')

            sel_mask = [in_range(dst.R,
                                 s.rmin,
                                 s.rmax).values & in_range(phirad_to_deg(dst.Phi),
                                                           s.phimin,
                                                           s.phimax).values for s in rps]
            MSK[i] = sel_mask

        return MSK

    logging.debug(f' --select_rphi_sectors:')
    MSK = selection_mask_rphi_sectors(dst, RPS)

    logging.debug(f' selection_mask computed, filling r-phi sectors')

    RGES = {}
    for i, msk in MSK.items():
        logging.debug(f' defining kr_event for sector {i}')
        RGES[i] = [kr_event(dst, dt, E, Q, sel_mask = m) for m in msk]

    return RGES


def select_xy_sectors(dst        : DataFrame,
                      time_diffs : np.array,
                      E          : np.array,
                      Q          : np.array,
                      bins_x     : np.array,
                      bins_y     : np.array)-> Dict[int, List[KrEvent]]:
    """
    Return a dict of KrEvent organized by xy sector

    Parameters
    ----------
        dst:
        The input data frame.
        time_diffs:
        An array of time differences needed to compute the time masks.
        E:
        An energy vector (can contain the corrected energy in the PMTs).
        Q:
        An energy vector (can contain the corrected energy in the SiPMs).
        bins_x:
        An array of bins along x.
        bins_y:
        An array of bins along y.

    Returns
    -------
        A map of selections defined as Dict[int, List[KrEvent]]
        where for each x (the key in the dict) one has a list
        (corresponding to y cells) of KrEvent (the events selected)

    """


    def selection_mask_xy_sectors(dst     : DataFrame,
                                  bins_x  : np.array,
                                  bins_y  : np.array)->Dict[int, np.array]:
        """Returns a dict of selections arranged in a dict of xy bins"""

        MSK = {}
        nbins_x = len(bins_x) -1
        nbins_y = len(bins_y) -1
        for i in range(nbins_x):
            logging.debug(f'computing selection mask for sector {i}')

            sel_x = in_range(dst.X.values, *bins_x[i: i+2])
            MSK[i] = [sel_x & in_range(dst.Y.values, *bins_y[j: j+2]) for j in range(nbins_y) ]

        return MSK

    logging.debug(f' function: select_xy_sectors')
    logging.debug(f' calling selection_mask')

    MSK = selection_mask_xy_sectors(dst, bins_x, bins_y)
    logging.debug(f' selection mask computed, filling selections')

    RGES = {}
    for i, msk in MSK.items():
        logging.debug(f' defining kr_event for sector {i}')
        RGES[i] = [kr_event(dst, time_diffs, E, Q, sel_mask = m) for m in msk]

    logging.debug(f' RGES computed')
    return RGES


def events_sector(nMap : Dict[int, List[float]])->np.array:
    N = []
    for  nL in nMap.values():
        N.append(np.mean(nL))
    return np.array(N)


def event_map(KRES : Dict[int, List[KrEvent]])->DataFrame:
    """
    Return an event map containing the events in each RPHI sector.

    Parameters
    ----------
        KRES:
        A map of selections (a dictionary of KrEvent).

    Returns
    -------
        A DataFrame containing the events in each RPHI sector.

    """
    nMap = {}
    for i, kres in KRES.items():
        nMap[i] = [len(k.S2e) for k in kres]
    return pd.DataFrame.from_dict(nMap)


def select_in_XYRange(kre : KrEvent, xyr : Range)->KrEvent:
    """ Selects a KrEvent in  a range of XY values"""

    sel  = in_range(kre.X, *xyr) & in_range(kre.Y, *xyr)

    return KrEvent(X    = kre.X[sel],
                   Y    = kre.Y[sel],
                   Z    = kre.Z[sel],
                   R    = kre.R[sel],
                   Phi  = kre.Phi[sel],
                   S2e  = kre.S2e[sel],
                   S1e  = kre.S1e[sel],
                   S2q  = kre.S2q[sel],
                   T    = kre.T[sel])



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
