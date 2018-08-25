import numpy as np
import matplotlib.dates  as md
import datetime
import matplotlib.pyplot as plt
from   invisible_cities.evm  .ic_containers  import Measurement
from   invisible_cities.core.system_of_units_c import units

from   invisible_cities.core.core_functions import loc_elem_1d
import invisible_cities.core.fit_functions as fitf

from . stat_functions       import mean_and_std
from . core_functions       import divide_np_arrays

from typing                 import Tuple
from . kr_types             import S1D, S2D
from pandas.core.frame      import DataFrame
from . kr_types             import Number, Int, Range
from typing                 import List, Tuple, Sequence, Iterable
from . kr_types             import KrEvent


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

def energy_time_profile(T           : np.array,
                        E           : np.array,
                        Tnbins      : int,
                        Trange      : Tuple[float, float],
                        timeStamps  : List[datetime.datetime],
                        erange      : Tuple[float, float] = (9e+3, 14e+3),
                        figsize     : Tuple[float, float] = (10,8)):

    xfmt = md.DateFormatter('%d-%m %H:%M')
    fig = plt.figure(figsize=figsize)

    x, y, yu = fitf.profileX(T, E, Tnbins, Trange)
    ax = fig.add_subplot(1, 1, 1)

    ax.xaxis.set_major_formatter(xfmt)
    plt.errorbar(timeStamps, y, yu, fmt="kp", ms=7, lw=3)
    plt.xlabel('date')
    plt.ylabel('E (pes)')
    plt.ylim(erange)
    plt.xticks( rotation=25 )


def energy_X_profile(X      : np.array,
                     E      : np.array,
                     xnbins : int,
                     xrange : Tuple[float, float],
                     xlabel : str = 'R',
                     erange : Tuple[float, float] = (9e+3, 14e+3),
                     figsize : Tuple[float, float] = (10,8)):

    fig = plt.figure(figsize=figsize)

    x, y, yu = fitf.profileX(X, E, xnbins, xrange)
    ax = fig.add_subplot(1, 1, 1)

    plt.errorbar(x, y, yu, fmt="kp", ms=7, lw=3)
    plt.xlabel(xlabel)
    plt.ylabel('E (pes)')
    plt.ylim(erange)
    

def s1d_from_dst(dst       : DataFrame,
                 range_s1e : Tuple[float, float] = (0,40),
                 range_s1w : Tuple[float, float] = (0,500),
                 range_s1h : Tuple[float, float] = (0,10),
                 range_s1t : Tuple[float, float] = (0,600))->S1D:

    hr = divide_np_arrays(dst.S1h.values, dst.S1e.values)
    return S1D(E = Measurement(*mean_and_std(dst.S1e.values,range_s1e)),
               W = Measurement(*mean_and_std(dst.S1w.values,range_s1w)),
               H = Measurement(*mean_and_std(dst.S1h.values,range_s1h)),
               R = Measurement(*mean_and_std(hr,(0,1))),
               T = Measurement(*mean_and_std(dst.S1t.values,range_s1t)))


def s2d_from_dst(dst : DataFrame)->S2D:
    return S2D(E = Measurement(*mean_and_std(dst.S2e.values,(0,20000))),
               W = Measurement(*mean_and_std(dst.S2w.values,(0,30))),
               Q = Measurement(*mean_and_std(dst.S2q.values,(0,1000))),
               N = Measurement(*mean_and_std(dst.Nsipm.values,(0,40))),
               X = Measurement(*mean_and_std(dst.X.values,(-200,200))),
               Y = Measurement(*mean_and_std(dst.Y.values,(-200,200))))




def plot_s1histos(dst, s1d, bins=20,
                  range_s1e = (0,40),
                  range_s1w = (0,500),
                  range_s1h = (0,10),
                  range_s1t = (0,600),
                  figsize=(12,12)):

    fig = plt.figure(figsize=figsize) # Creates a new figure
    ax = fig.add_subplot(3, 2, 1)

    ax.set_xlabel('S1 energy (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)#ylabel
    ax.hist(dst.S1e,
            range=range_s1e,
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.E.value, s1d.E.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 2)
    ax.set_xlabel(r'S1 width ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(dst.S1w,
            range=range_s1w,
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.W.value, s1d.W.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 3)
    ax.set_xlabel(r'S1 height (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(dst.S1h,
            range=range_s1h,
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.H.value, s1d.H.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 4)
    hr = divide_np_arrays(dst.S1h.values, dst.S1e.values)

    ax.set_xlabel(r'height / energy ',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(hr,
            range=(0,1),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.R.value, s1d.R.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 5)
    ax.set_xlabel(r'S1 time ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('Frequency', fontsize = 11)
    ax.hist(dst.S1t / units.mus,
            range=range_s1t,
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s1d.R.value, s1d.R.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 6)
    plt.hist2d(dst.S1t/units.mus, dst.S1e, bins=10, range=(range_s1t,range_s1e))
    plt.colorbar()
    ax.set_xlabel(r'S1 time ($\mu$s) ',fontsize = 11) #xlabel
    ax.set_ylabel('S1 height (pes)', fontsize = 11)
    plt.grid(True)

    plt.tight_layout()


def plot_s2histos(df, s2d, bins=20, emin=3000, emax=15000, figsize=(12,12)):

    fig = plt.figure(figsize=figsize) # Creates a new figure
    ax = fig.add_subplot(3, 2, 1)

    ax.set_xlabel('S2 energy (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('# events', fontsize = 11)#ylabel
    ax.hist(df.S2e,
            range=(emin, emax),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.E.value, s2d.E.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 2)

    ax.set_xlabel(r'S2 width ($\mu$s)',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.S2w,
            range=(0,30),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.W.value, s2d.W.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 3)

    ax.set_xlabel(r'Q (pes)',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.S2q,
            range=(0,1000),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.Q.value, s2d.Q.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 4)

    ax.set_xlabel(r'number SiPM',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.Nsipm,
            range=(0,30),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.N.value, s2d.N.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 5)

    ax.set_xlabel(r' X (mm)',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.X,
            range=(-200,200),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.X.value, s2d.X.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    ax = fig.add_subplot(3, 2, 6)
    ax.set_xlabel(r' Y (mm)',fontsize = 11) #xlabel
    ax.set_ylabel('# of events', fontsize = 11)
    ax.hist(df.Y,
            range=(-200,200),
            bins=bins,
            histtype='step',
            edgecolor='black',
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(s2d.Y.value, s2d.Y.uncertainty))
    ax.legend(fontsize= 10, loc='upper right')
    plt.grid(True)

    plt.tight_layout()
