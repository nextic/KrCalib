"""Module ranges_and_bins_functions.
This module includes the functions related with ranges and bins.

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

from   invisible_cities.icaro. hst_functions import shift_to_bin_centers

from . kr_types             import Number, Int, Range, Array
from . kr_types             import KrBins, KrNBins, KrRanges, KrTimes
from typing                 import List, Tuple, Sequence, Iterable, Dict


import sys
import logging
log = logging.getLogger()


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


    T      = dst.time.values   # vector of times, will be all zeros if MC
    if (T==0).all() : # MC!
        tstart   = 0                         # time linear on events
        tfinal   = len(T)
    else:
        dst_time = dst.sort_values('time')  # if data, sort on time,
        T        = dst_time.time.values     # event number can be repeated with multiple DSTs
        tstart   = T[0]
        tfinal   = T[-1]

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
