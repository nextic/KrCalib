import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Any

from . core_functions  import mean_and_std
from . core_functions  import Number
from . kr_types        import PlotLabels


def labels(pl : PlotLabels):
    """
    Set x and y labels.
    """
    plt.xlabel(pl.x)
    plt.ylabel(pl.y)
    plt.title (pl.title)

def h1d(x      : np.array,
       bins    : int,
       range   : Tuple[float],
       weights : Any   = None,
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


def display_h1d(x      : np.array,
                bins    : int,
                range   : Tuple[float],
                weights : Any   = None,
                log     : bool  = False,
                normed  : bool  = False,
                color   : str   = 'black',
                width   : float = 1.5,
                style   : str   ='solid',
                pltLabels=PlotLabels(x='x', y='y', title=None),
                figsize=(6,6)):

    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 1, 1)
    n, b, p = h1d(x, bins=bins, range = range)
    plot_histo(pltLabels, ax)

def gaussian_histo_example(mean: float, nevt: float = 1e5):
    Nevt  = int(nevt)
    sigmas = np.random.uniform(low=1.0, high=10., size=4)

    fig = plt.figure(figsize=(10,10))
    pltLabels =PlotLabels(x='Energy', y='Events', title='Gaussian')

    e = np.random.normal(100, sigmas[0], Nevt)
    ax      = fig.add_subplot(2, 2, 1)
    n, b, p = h1d(e, bins=100, range=(mean - 5 * sigmas[0],mean + 5 * sigmas[0]))
    plot_histo(pltLabels, ax)

    e = np.random.normal(100, sigmas[1], Nevt)
    ax      = fig.add_subplot(2, 2, 2)
    n, b, p = h1d(e, bins=100, range=(mean - 5 * sigmas[1],mean + 5 * sigmas[1]), log=True)
    pltLabels.title = 'Gaussian log scale'
    plot_histo(pltLabels, ax, legendloc='upper left')

    e = np.random.normal(100, sigmas[2], Nevt)
    ax      = fig.add_subplot(2, 2, 3)
    n, b, p = h1d(e, bins=100, range=(mean - 5 * sigmas[2],mean + 5 * sigmas[2]), normed=True)
    pltLabels.title = 'Gaussian normalized'
    plot_histo(pltLabels, ax, legendloc='upper right')

    e = np.random.normal(100, sigmas[3], Nevt)
    ax      = fig.add_subplot(2, 2, 4)
    n, b, p = h1d(e, bins=100, range=(mean - 5 * sigmas[3],mean + 5 * sigmas[3]),
                  color='red', width=2.0, style='dashed')
    pltLabels.title = 'Gaussian change histo pars'
    plot_histo(pltLabels, ax, legendsize=14)

    plt.tight_layout()
