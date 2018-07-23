import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing      import List, Tuple
import warnings

#import matplotlib.dates  as md
from   invisible_cities.core.core_functions import in_range
import invisible_cities.core.fit_functions as fitf

#from . fit_functions import gauss_seed
from . fit_functions import chi2

from invisible_cities.core .stat_functions import poisson_sigma
from invisible_cities.icaro. hst_functions import shift_to_bin_centers
from invisible_cities.types.ic_types       import NN

from . kr_types import GaussPar
from . kr_types import FitPar
from . kr_types import HistoPar
from . kr_types import FitCollection
from krcal.kr_types import PlotLabels

from . core_functions import mean_and_std
from . core_functions import labels

def gaussian_fit(x       : np.array,
                 y       : np.array,
                 seed    : GaussPar,
                 n_sigma : int):
    """Gaussian fit to x,y variables, with fit range defined by n_sigma"""

    #seed      = gauss_seed(x, y)
    #fit_range = seed[1] - n_sigma * seed[2], seed[1] + n_sigma * seed[2]

    fit_range = seed.mu - n_sigma * seed.std, seed.mu + n_sigma * seed.std

    x, y      = x[in_range(x, *fit_range)], y[in_range(x, *fit_range)]
    yu        = poisson_sigma(y)
    fseed     =(seed.amp, seed.mu, seed.std)
    try:
        f     = fitf.fit(fitf.gauss, x, y, fseed, sigma=yu)
        c2    = chi2(f, x, y, yu)
        valid = True
    except RuntimeError:
        warnings.warn(f' fit failed for seed  = {seed} ', UserWarning)
        valid = False
        c2 = NN
    return FitPar(x  = x,
                  y  = y,
                  yu = yu,
                  f  = f,
                  chi2 = c2,
                  valid = valid)


def energy_fit(e : np.array,
               nbins   : int,
               range   : Tuple[float],
               n_sigma : float = 2.5)->FitCollection:
    """
    Takes an "energy vector" (e.g, 1d array), with number of bins enbins and range erange, then:
        1. Computes the histogram of e with enbins in erange. This returns an array of bin
        edges (b), and bin contents (y). The array (b) is shifted to bin centers (x)
        2. The arrays x and y are fitted to a gaussian, in a range given by an interval
        arround the estimation of the maximum of the gaussian. The interval size is estimated
        by multiplying n_sigma by the estimation of the gaussian std.

    The result of the fit is a fit collection, that includes a FitPar and a HistoPar objects
    needed for printing and plotting the fit result.
       """

    y, b = np.histogram(e, bins= nbins, range=range)
    x = shift_to_bin_centers(b)

    mu, std = mean_and_std(e, range)
    amp     = mu * (2 * np.pi)**0.5 * std * np.diff(x)[0]
    seed = GaussPar(mu = mu, std = std, amp = amp)

    fp = gaussian_fit(x, y, seed, n_sigma)

    hp = HistoPar(var      = e,
                  nbins    = nbins,
                  range    = range)

    return FitCollection(fp = fp, hp = hp, seed = seed)


def plot_energy_fit(fc : FitCollection):
    """Takes a KrEvent and a FitPar object and plots fit"""


    par  = np.array(fc.fp.f.values)
    err  = np.array(fc.fp.f.errors)
    _, _, _   = plt.hist(fc.hp.var,
                         bins = fc.hp.nbins,
                         range=fc.hp.range,
                         histtype='step',
                         edgecolor='black',
                         linewidth=1.5,
                         label=r'$\mu={:7.2f} +- {:7.3f},\ \sigma={:7.2f} +- {:7.3f}$'.format(
                               par[1], err[1], par[2], err[2]))

    plt.plot(fc.fp.x, fc.fp.f.fn(fc.fp.x), "r-", lw=4)


def display_energy_fit(fc : FitCollection, figsize : Tuple[int] =(6,6), legend_loc='best'):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    plot_energy_fit(fc)
    ax.legend(fontsize= 10, loc=legend_loc)


def print_energy_fit(fc : FitCollection):

    par  = np.array(fc.fp.f.values)
    err  = np.array(fc.fp.f.errors)
    try:
        r  = 2.35 * 100 *  par[2] / par[1]
        fe = np.sqrt(41 / 2458) * r

        print(f' Emu       = {par[1]} +-{err[1]} ')
        print(f' E sigma   = {par[2]} +-{err[2]} ')
        print(f' chi2    = {fc.fp.chi2} ')

        print(f' sigma E/E (FWHM)     (%) ={r}')
        print(f' sigma E/E (FWHM) Qbb (%) ={fe} ')
    except ZeroDivisionError:
        warnings.warn(f' mu  = {par[1]} ', UserWarning)


def plot_energy_fit_chi2(fc : FitCollection):
    """Takes a KrEvent and a FitPar object and plots fit"""

    x  = fc.fp.x
    f  = fc.fp.f
    y  = fc.fp.y
    yu = fc.fp.yu
    plt.errorbar(x, (f.fn(x) - y) / yu, 1, np.diff(x)[0] / 2, fmt="p", c="k")
    lims = plt.xlim()
    plt.plot(lims, (0, 0), "g--")
    plt.xlim(*lims)
    plt.ylim(-5, +5)


def display_energy_fit_and_chi2(fc : FitCollection, pl : PlotLabels, figsize : Tuple[int] =(6,6),
                                legend_loc : str = 'best'):
    fig = plt.figure(figsize=figsize)
    #ax = fig.add_subplot(1, 1, 1)
    #ax.legend(fontsize= 10, loc=legend_loc)
    frame_data = plt.gcf().add_axes((.1, .3,.8, .6))
    plot_energy_fit(fc)
    labels(pl)
    frame_res = plt.gcf().add_axes((.1, .1,
                                 .8, .2))
    frame_data.set_xticklabels([])

    plot_energy_fit_chi2(fc)




#
#
# def energy_fit_in_XYRange(kre    : KrEvent,
#                           enbins : int,
#                           erange : Tuple[float],
#                           xr     : Tuple[float],
#                           yr     : Tuple[float])->FitCollection:
#
#     sel  = in_range(kre.X, *xr) & in_range(kre.Y, *yr)
#     e    = kre.E[sel]
#
#     return energy_fit(e, enbins, erange)
#
#
# def plot_energy_fit_in_XYRange(fc : FitCollection):
#
#     frame_data = plt.gcf().add_axes((.1, .3,.8, .6))
#     plot_energy_fit(fc)
#
#     frame_data.set_xticklabels([])
#     labels("", "Entries", "Energy fit ")
#     frame_res = plt.gcf().add_axes((.1, .1,
#                                 .8, .2))
#     plot_energy_chi2(fc)
#
#
# def energy_fits_in_fiducial_regions(kdst   : DstEvent,
#                                     enbins : int,
#                                     erange : Tuple[float]):
#     """Energy fits in fiducial regions"""
#
#     fc_full  = energy_fit(kdst.full.E, enbins=enbins, erange=erange)
#     fc_fid   = energy_fit(kdst.fid.E, enbins=enbins, erange=erange)
#     fc_core  = energy_fit(kdst.core.E, enbins=enbins, erange=erange)
#     fc_hcore = energy_fit(kdst.hcore.E, enbins=enbins, erange=erange)
#
#     return FitCollections(full  = fc_full,
#                           fid   = fc_fid,
#                           core  = fc_core,
#                           hcore = fc_hcore)
#
#
# def plot_energy_fits_in_fiducial_regions(fc : FitCollections):
#     """Plot energy fits in fiducial regions"""
#
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(2, 2, 1)
#     plot_energy_fit(fc.full)
#     l = ax.legend(fontsize= 10, loc='upper right')
#     ax = fig.add_subplot(2, 2, 2)
#     plot_energy_fit(fc.fid)
#     l = ax.legend(fontsize= 10, loc='upper right')
#     ax = fig.add_subplot(2, 2, 3)
#     plot_energy_fit(fc.core)
#     l = ax.legend(fontsize= 10, loc='upper right')
#     ax = fig.add_subplot(2, 2, 4)
#     l = plot_energy_fit(fc.hcore)
#     ax.legend(fontsize= 10, loc='upper right')
#
#
# def print_energy_fits_in_fiducial_regions(fc : FitCollections):
#     """Plot energy fits in fiducial regions"""
#
#     print_energy_fit(fc.full)
#     print_energy_fit(fc.fid)
#     print_energy_fit(fc.core)
#     print_energy_fit(fc.hcore)
