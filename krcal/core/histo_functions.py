import numpy as np
import random

import matplotlib.pyplot as plt

from typing import Tuple

from . stat_functions  import mean_and_std
from . kr_types import Number, Array
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
       style   : str   ='solid'):
    """
    histogram 1d with continuous steps and display of statsself.
    number of bins (bins) and range are compulsory.
    """

    mu, std = mean_and_std(x, range)
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
                       label     = r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, std))

    return n, b, p


def plot_histo(pltLabels: PlotLabels, ax, legendsize=10, legendloc='best', labelsize=11):
    ax.legend(fontsize= legendsize, loc=legendloc)
    ax.set_xlabel(pltLabels.x,fontsize = labelsize)
    ax.set_ylabel(pltLabels.y, fontsize = labelsize)
    if pltLabels.title:
        plt.title(pltLabels.title)


def h1d(x       : np.array,
        bins    : int,
        range   : Tuple[float],
        weights : Array = None,
        log     : bool  = False,
        normed  : bool  = False,
        color   : str   = 'black',
        width   : float = 1.5,
        style   : str   ='solid',
        pltLabels=PlotLabels(x='x', y='y', title=None),
        figsize=(6,6)):

    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 1, 1)
    n, b, p = h1(x, bins=bins, range = range)
    plot_histo(pltLabels, ax)
