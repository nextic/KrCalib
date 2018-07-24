import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . core_functions  import mean_and_std

def h1d(x, bins=None, range=None, weights=None, log=False, normed=False,
        pltLabels='None', edgecolor='black',
        legend = 'best', figsize=(6,6)):
    """histogram 1d with continuous steps and display of stats"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    mu, std = mean_and_std(x, range)
    ax.set_xlabel(pltLabels.x,fontsize = 11)
    ax.set_ylabel(pltLabels.y, fontsize = 11)

    ax.hist(x,
            bins= bins,
            range=range,
            weights=weights,
            log = log,
            normed = normed,
            histtype='step',
            edgecolor=edgecolor,
            linewidth=1.5,
            label=r'$\mu={:7.2f},\ \sigma={:7.2f}$'.format(mu, std))
    ax.legend(fontsize= 10, loc=legend)
    plt.grid(True)

    if pltLabels.title:
        plt.title(pltLabels.title)

    return mu, std
