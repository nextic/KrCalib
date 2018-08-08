import numpy as np
import random

import matplotlib.pyplot as plt
from . import fit_functions_ic as fitf

from typing import Tuple

from . stat_functions  import mean_and_std
from . kr_types import Number, Array, Str
from . kr_types        import PlotLabels


def labels(pl : PlotLabels):
    """
    Set x and y labels.
    """
    plt.xlabel(pl.x)
    plt.ylabel(pl.y)
    plt.title (pl.title)

def h1(x      : np.array,
       bins    : int,
       range   : Tuple[float],
       weights : Array = None,
       log     : bool  = False,
       normed  : bool  = False,
       color   : str   = 'black',
       width   : float = 1.5,
       style   : str   ='solid',
       stats   : bool  = True,
       lbl     : Str   = None):
    """
    histogram 1d with continuous steps and display of statsself.
    number of bins (bins) and range are compulsory.
    """

    mu, std = mean_and_std(x, range)

    if stats:
        entries  =  f'Entries = {len(x)}'
        mean     =  r'$\mu$ = {:7.2f}'.format(mu)
        sigma    =  r'$\sigma$ = {:7.2f}'.format(std)
        stat     =  f'{entries}\n{mean}\n{sigma}'
    else:
        stat     = ''

    if lbl == None:
        lab = ''
    else:
        lab = lbl

    lab = stat + lab

    n, b, p = plt.hist(x,
                       bins      = bins,
                       range     = range,
                       weights   = weights,
                       log       = log,
                       density   = normed,
                       histtype  = 'step',
                       edgecolor = color,
                       linewidth = width,
                       linestyle = style,
                       label     = lab)

    return n, b


def plot_histo(pltLabels: PlotLabels, ax, legendsize=10, legendloc='best', labelsize=11):
    ax.legend(fontsize= legendsize, loc=legendloc)
    ax.set_xlabel(pltLabels.x,fontsize = labelsize)
    ax.set_ylabel(pltLabels.y, fontsize = labelsize)
    if pltLabels.title:
        plt.title(pltLabels.title)


def h1d(x         : np.array,
        bins      : int,
        range     : Tuple[float],
        weights   : Array               = None,
        log       : bool                = False,
        normed    : bool                = False,
        color     : str                 = 'black',
        width     : float               = 1.5,
        style     : str                 ='solid',
        stats     : bool                = True,
        lbl       : Str                 = None,
        pltLabels : PlotLabels          =PlotLabels(x='x', y='y', title=None),
        legendloc : str                 ='best',
        figsize   : Tuple[float, float] =(6,6)):

    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 1, 1)
    n, b    = h1(x, bins=bins, range = range, stats = stats, lbl = lbl)
    plot_histo(pltLabels, ax, legendloc=legendloc)
    return n, b


def h2d(x       : np.array,
        y       : np.array,
        nbins_x : int,
        nbins_y : int,
        range_x : Tuple[float],
        range_y : Tuple[float],
        pltLabels=PlotLabels(x='x', y='y', title=None),
        figsize=(10,6)):

    xbins  = np.linspace(*range_x, nbins_x + 1)
    ybins  = np.linspace(*range_y, nbins_y + 1)
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)
    nevt, *_  = plt.hist2d(x, y, (xbins, ybins))
    plt.colorbar().set_label("Number of events")
    labels(pltLabels)
    return nevt

def h2d(x         : np.array,
        y         : np.array,
        nbins_x   : int,
        nbins_y   : int,
        range_x   : Tuple[float],
        range_y   : Tuple[float],
        pltLabels : PlotLabels   = PlotLabels(x='x', y='y', title=None),
        profile  : bool          = True,
        figsize=(10,6)):

    xbins  = np.linspace(*range_x, nbins_x + 1)
    ybins  = np.linspace(*range_y, nbins_y + 1)

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)
    nevt, *_  = plt.hist2d(x, y, (xbins, ybins))
    plt.colorbar().set_label("Number of events")

    if profile:
        x, y, yu     = fitf.profileX(x, y, nbins_x)
        valid_points = ~np.isnan(yu)
        x    = x [valid_points]
        y    = y [valid_points]
        yu   = yu[valid_points]
        plt.errorbar(x, y, yu, np.diff(x)[0]/2, fmt="kp", ms=7, lw=3)

    labels(pltLabels)
    return nevt
