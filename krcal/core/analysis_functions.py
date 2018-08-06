import numpy as np
import datetime
from   pandas.core.frame import DataFrame
import matplotlib.dates  as md
import matplotlib.pyplot as plt
import invisible_cities.core.fit_functions as fitf
from invisible_cities.icaro. hst_functions import shift_to_bin_centers
from   invisible_cities.core.core_functions import in_range

from   typing      import Tuple, List, Iterable
from . kr_types    import Number, Int, Range
from . kr_types    import KrBins, KrNBins, KrRanges, KrTimes, KrEvent


def s12_time_profile(kdst       : KrEvent,
                     Tnbins     : Int,
                     Trange     : Range,
                     timeStamps : List[datetime.datetime],
                     s2range    : Range =(8e+3, 1e+4),
                     s1range    : Range =(10,11),
                     figsize=(8,8)):

    xfmt = md.DateFormatter('%d-%m %H:%M')
    fig = plt.figure(figsize=figsize)

    x, y, yu = fitf.profileX(kdst.T, kdst.S2e, Tnbins, Trange)
    ax = fig.add_subplot(1, 2, 1)
    #plt.figure()
    #ax=plt.gca()
    #fig.add_subplot(1, 2, 1)
    ax.xaxis.set_major_formatter(xfmt)
    plt.errorbar(timeStamps, y, yu, fmt="kp", ms=7, lw=3)
    plt.xlabel('date')
    plt.ylabel('S2 (pes)')
    plt.ylim(s2range)
    plt.xticks( rotation=25 )

    x, y, yu = fitf.profileX(kdst.T, kdst.S1e, Tnbins, Trange)
    ax = fig.add_subplot(1, 2, 2)
    #ax=plt.gca()

    #xfmt = md.DateFormatter('%d-%m %H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    plt.errorbar(timeStamps, y, yu, fmt="kp", ms=7, lw=3)
    plt.xlabel('date')
    plt.ylabel('S1 (pes)')
    plt.ylim(s1range)
    plt.xticks( rotation=25 )
    plt.tight_layout()

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



def kr_event(dst : DataFrame)->KrEvent:
    dst_time = dst.sort_values('event')
    return KrEvent(X   = dst.X.values,
                   Y   = dst.Y.values,
                   Z   = dst.Z.values,
                   R   = dst.R.values,
                   Phi = dst.R.values,
                   T   = dst_time.time.values,
                   S2e = dst.S2e.values,
                   S1e = dst.S1e.values,
                   S2q = dst.S2q.values)


def fiducial_volumes(dst     : DataFrame,
                     R_full  : float  = 200,
                     R_fid   : float  = 150,
                     R_core  : float  = 100,
                     R_hcore : float  = 50)->Iterable[KrEvent]:


    dst_full   = dst[dst.R < R_full]
    dst_fid    = dst[dst.R < R_fid]
    dst_core   = dst[dst.R < R_core]
    dst_hcore  = dst[dst.R < R_hcore]

    n_dst      = len(dst)
    n_full     = len(dst_full)
    n_fid      = len(dst_fid)
    n_core     = len(dst_core)
    n_hcore    = len(dst_hcore)

    eff_full   = n_full  / n_dst
    eff_fid    = n_fid   / n_dst
    eff_core   = n_core  / n_dst
    eff_hcore  = n_hcore / n_dst

    print(f" nfull : {n_full}: eff_full = {eff_full} ")
    print(f" nfid : {n_fid}: eff_fid = {eff_fid} ")
    print(f" ncore : {n_core}: eff_core = {eff_core} ")
    print(f" nhcore : {n_hcore}: eff_hcore = {eff_hcore} ")

    return kr_event(dst_full), kr_event(dst_fid), kr_event(dst_core), kr_event(dst_hcore)


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
