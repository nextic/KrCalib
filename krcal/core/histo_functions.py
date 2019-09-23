import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
from . import fit_functions_ic as fitf

from typing import Tuple, Optional

from . stat_functions  import mean_and_std
from . kr_types import Number, Array, Str
from . kr_types        import PlotLabels

from  invisible_cities.core.core_functions import shift_to_bin_centers


def labels(pl : PlotLabels):
    """
    Set x and y labels.
    """
    plt.xlabel(pl.x)
    plt.ylabel(pl.y)
    plt.title (pl.title)


def profile1d(z : np.array,
              e : np.array,
              nbins_z : int,
              range_z : np.array)->Tuple[float, float, float]:
    """Adds an extra layer to profileX, returning only valid points"""
    x, y, yu     = fitf.profileX(z, e, nbins_z, range_z)
    valid_points = ~np.isnan(yu)
    x    = x [valid_points]
    y    = y [valid_points]
    yu   = yu[valid_points]
    return x, y, yu


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
       lbl     : Optional[str]  = None):
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
        lab = ' '
    else:
        lab = lbl

    lab = stat + lab

    if color == None:
        n, b, p = plt.hist(x,
                       bins      = bins,
                       range     = range,
                       weights   = weights,
                       log       = log,
                       density   = normed,
                       histtype  = 'step',
                       linewidth = width,
                       linestyle = style,
                       label     = lab)

    else:

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

    return n, b, mu, std


def plot_histo(pltLabels: PlotLabels, ax, legend= True,
               legendsize=10, legendloc='best', labelsize=11):

    if legend:
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
    n, b, mu, std    = h1(x, bins=bins, range = range, stats = stats, lbl = lbl,
                          normed = normed, color = color,
                          width = width, style=style,
                          weights=weights, log=log)
    plot_histo(pltLabels, ax, legendloc=legendloc)
    return n, b, mu, std


def h2(x         : np.array,
       y         : np.array,
       nbins_x   : int,
       nbins_y   : int,
       range_x   : Tuple[float],
       range_y   : Tuple[float],
       profile   : bool   = True):

    xbins  = np.linspace(*range_x, nbins_x + 1)
    ybins  = np.linspace(*range_y, nbins_y + 1)

    nevt, *_  = plt.hist2d(x, y, (xbins, ybins))
    plt.colorbar().set_label("Number of events")

    if profile:
        x, y, yu     = profile1d(x, y, nbins_x, range_x)
        plt.errorbar(x, y, yu, np.diff(x)[0]/2, fmt="kp", ms=7, lw=3)

    return nevt


def h2d(x         : np.array,
        y         : np.array,
        nbins_x   : int,
        nbins_y   : int,
        range_x   : Tuple[float],
        range_y   : Tuple[float],
        pltLabels : PlotLabels   = PlotLabels(x='x', y='y', title=None),
        profile  : bool          = False,
        figsize=(10,6)):

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)

    nevt   = h2(x, y, nbins_x, nbins_y, range_x, range_y, profile)
    labels(pltLabels)
    return nevt


def compute_similar_histo(param     : np.array,
                          reference : pd.DataFrame
                          )-> Tuple[np.array, np.array]:
    """
    This function computes a histogram with the same
    bin_size and number of bins as a given one.
    Parameters
    ----------
    param : np.array
        Array to be represented in the histogram.
    reference: pd.DataFrame
        Dataframe with the information of a reference histogram.
    Returns
    ----------
        Two arrays with the entries and the limits of each bin.
    """
    bin_size   = np.diff(reference.bin_centres)[0]
    min_Z_hist = reference.bin_centres.values[ 0] - bin_size/2
    max_Z_hist = reference.bin_centres.values[-1] + bin_size/2
    N, b = np.histogram(param, bins = len(reference),
                        range =(min_Z_hist, max_Z_hist));
    return N, b


def normalize_histo_and_poisson_error(N : np.array,
                                      b : np.array
                                      )->Tuple[np.array,
                                               np.array]:
    """
    Computes poissonian error for each bin. Normalizes the histogram
    with its area, applying this factor also to the error.
    Parameters
    ----------
    N: np.array
        Array with the entries inside each bin.
    b: np.array
        Array with limits of each bin.
    Returns
    ----------
        The input N array normalized, and its error multiplied
        by same normalization value.
    """
    err_N = np.sqrt(N)

    norm  = 1/sum(N)/((b[-1]-b[0])/(len(b)-1))
    N     = N*norm
    err_N = err_N*norm

    return N, err_N


def compute_and_save_hist_as_pd(values     : np.array           ,
                                out_file   : pd.HDFStore        ,
                                hist_name  : str                ,
                                n_bins     : int                ,
                                range_hist : Tuple[float, float],
                                norm       : bool               )->None:
    """
    Computes 1d-histogram and saves it in a file.
    The name of the table inside the file must be provided.
    Parameters
    ----------
    values : np.array
        Array with values to be plotted.
    out_file: pd.HDFStore
        File where histogram will be saved.
    hist_name: string
        Name of the pd.Dataframe to contain the histogram.
    n_bins: int
        Number of bins to make the histogram.
    range_hist: length-2 tuple (optional)
        Range of the histogram.
    norm: bool
        If True, histogram will be normalized.
    """
    n, b = np.histogram(values, bins = n_bins,
                        range = range_hist,
                        density = norm)
    table = pd.DataFrame({'entries': n,
                          'magnitude': shift_to_bin_centers(b)})
    out_file.put(hist_name, table, format='table', data_columns=True)

    return;
