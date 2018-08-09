import numpy as np
import random

import matplotlib.pyplot as plt
from . histo_functions import h1
from . histo_functions import plot_histo
from . kr_types        import PlotLabels


def gaussian_histo_example(mean, nevt, figsize=(10,10)):
    """Examples of histogramming gaussian distributions"""

    Nevt   = int(nevt)
    sigmas = np.random.uniform(low=1.0, high=10., size=4)

    fig       = plt.figure(figsize=figsize)
    pltLabels = PlotLabels(x='Energy', y='Events', title='Gaussian')

    e   = np.random.normal(100, sigmas[0], Nevt)
    ax  = fig.add_subplot(2, 2, 1)
    (_) = h1(e, bins=100, range=(mean - 5 * sigmas[0],mean + 5 * sigmas[0]))
    plot_histo(pltLabels, ax)

    pltLabels.title = 'Gaussian log scale'
    e   = np.random.normal(100, sigmas[1], Nevt)
    ax  = fig.add_subplot(2, 2, 2)
    (_) = h1(e, bins=100, range=(mean - 5 * sigmas[1],mean + 5 * sigmas[1]), log=True)
    plot_histo(pltLabels, ax, legendloc='upper left')

    pltLabels.title = 'Gaussian normalized'
    e    = np.random.normal(100, sigmas[2], Nevt)
    ax   = fig.add_subplot(2, 2, 3)
    (_)  = h1(e, bins=100, range=(mean - 5 * sigmas[2],mean + 5 * sigmas[2]), normed=True)
    plot_histo(pltLabels, ax, legendloc='upper right')

    pltLabels.title = 'Gaussian change histo pars'
    e    = np.random.normal(100, sigmas[3], Nevt)
    ax   = fig.add_subplot(2, 2, 4)
    (_)  = h1(e, bins=100, range=(mean - 5 * sigmas[3],mean + 5 * sigmas[3]),
                 color='red', width=2.0, style='dashed')
    plot_histo(pltLabels, ax, legendsize=14)

    plt.tight_layout()


def histo_gaussian_experiment_sample(exps, mexperiments, samples=9, canvas=(3,3),
                                     bins = 50, range_e = (9e+3,11e+3),
                                     figsize=(10,10)):
    """Takes an array of gaussian experiments and samples it"""
    fig = plt.figure(figsize=figsize)
    ks  = random.sample(range(mexperiments), samples)
    for i,k in enumerate(ks):
        ax  = fig.add_subplot(canvas[0], canvas[1], i+1)
        (_) = h1(exps[k], bins=bins, range = range_e)

    plt.tight_layout()


def histo_gaussian_params_and_pulls(mean, sigma, mus, umus, stds, ustds,
                                    bin_mus    = 50,
                                    bin_stds   = 50,
                                    bin_pull   = 50,
                                    range_mus  = (9950,10050),
                                    range_stds = (150,250),
                                    range_pull = (-5,5),
                                    figsize    =(10,10)):
    """Histograms mus and stds of gaussian experiments (values and pulls)"""
    fig       = plt.figure(figsize=figsize)

    ax        = fig.add_subplot(2, 2, 1)
    pltLabels = PlotLabels(x='mus', y='Events', title='mean')
    (_)       = h1(mus, bins=bin_mus, range=range_mus)
    plot_histo(pltLabels, ax)

    ax        = fig.add_subplot(2, 2, 2)
    (_)       = h1((mus-mean) / umus, bins=bin_pull, range=range_pull)
    pltLabels = PlotLabels(x='Pull(mean)', y='Events', title='Pull (mean)')
    plot_histo(pltLabels, ax)

    ax        = fig.add_subplot(2, 2, 3)
    pltLabels = PlotLabels(x='std ', y='Events', title='std')
    (_)       = h1(stds, bins=bin_stds, range=range_stds)
    plot_histo(pltLabels, ax)

    ax      = fig.add_subplot(2, 2, 4)
    n, b, p = h1((stds-sigma) / ustds, bins=50, range=range_pull)
    pltLabels =PlotLabels(x='pull (std) ', y='Events', title='Pull (std)')
    plot_histo(pltLabels, ax)

    plt.tight_layout()


def histo_lt_params_and_pulls(e0, lt, e0s, ue0s, lts, ults,
                              bin_e0s    = 50,
                              bin_lts    = 50,
                              bin_pull   = 50,
                              range_e0s  = (9950,10050),
                              range_lts  = (1900,2100),
                              range_pull = (-5,5),
                              figsize    =(10,10)):
    """Histogram mus and stds of LT experiments"""

    fig       = plt.figure(figsize=figsize)

    ax      = fig.add_subplot(2, 2, 1)
    pltLabels =PlotLabels(x='E0', y='Events', title='E0')
    (_)     = h1(e0s, bins=bin_e0s, range=range_e0s)
    plot_histo(pltLabels, ax)

    ax      = fig.add_subplot(2, 2, 2)
    (_)     = h1((e0s-e0) / ue0s, bins=bin_pull, range=range_pull)
    pltLabels =PlotLabels(x='Pull(E0)', y='Events', title='Pull (E0)')
    plot_histo(pltLabels, ax, legendloc='upper left')

    ax      = fig.add_subplot(2, 2, 3)
    pltLabels =PlotLabels(x='LT ', y='Events', title='LT')
    (_)     = h1(lts, bins=bin_lts, range=range_lts)
    plot_histo(pltLabels, ax)

    ax      = fig.add_subplot(2, 2, 4)
    (_)     = h1((lts-lt) / ults, bins=bin_pull, range=range_pull)
    pltLabels =PlotLabels(x='pull (LT) ', y='Events', title='Pull (LT)')
    plot_histo(pltLabels, ax, legendloc='upper left')

    plt.tight_layout()
