import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing      import List, Tuple
import warnings

#import matplotlib.dates  as md
from   invisible_cities.core.core_functions import in_range
#import invisible_cities.core.fit_functions as fitf
from . import fit_functions_ic as fitf

from   invisible_cities.evm  .ic_containers  import Measurement
from . fit_functions import chi2
from . core_functions import mean_and_std
from . core_functions import Number

from invisible_cities.core .stat_functions import poisson_sigma
from invisible_cities.icaro. hst_functions import shift_to_bin_centers
from invisible_cities.types.ic_types       import NN

from . kr_types import GaussPar
from . kr_types import FitPar
from . kr_types import FitResult
from . kr_types import HistoPar
from . kr_types import FitCollection
from . kr_types import PlotLabels

from . histo_functions import labels
from scipy.optimize import OptimizeWarning
from numpy import sqrt, pi


def gaussian_parameters(x : np.array, range : Tuple[Number], bin_size : float)->GaussPar:
    """
    Return the parameters defining a Gaussian
    g = N * exp(x - mu)**2 / (2 * std**2)
    where N is the normalization: N = 1 / (sqrt(2 * pi) * std)
    The parameters returned are the mean (mu), standard deviation (std)
    and the amplitude (inverse of N).
    """
    mu, std = mean_and_std(x, range)
    ff     = sqrt(2 * pi) * std
    amp     = len(x) * bin_size / ff

    sel  = in_range(x, *range)
    N = len(x[sel])              # number of samples in range
    mu_u  = std / sqrt(N)
    std_u = std / sqrt(2 * (N -1))
    amp_u = sqrt(2 * np.pi) * std_u

    return GaussPar(mu  = Measurement(mu, mu_u),
                    std = Measurement(std, std_u),
                    amp = Measurement(amp, amp_u))


def gaussian_fit(x       : np.array,
                 y       : np.array,
                 seed    : GaussPar,
                 n_sigma : int):
    """Gaussian fit to x,y variables, with fit range defined by n_sigma"""

    mu  = seed.mu.value
    std = seed.std.value
    amp = seed.amp.value

    fit_range = mu - n_sigma * std, mu + n_sigma * std

    x, y      = x[in_range(x, *fit_range)], y[in_range(x, *fit_range)]
    yu        = poisson_sigma(y)
    fseed     = (amp, mu, std)

    fp = None
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            # print(x)
            # print(y)
            # print(yu)
            # print(fseed)
            f     = fitf.fit(fitf.gauss, x, y, fseed, sigma=yu)
            c2    = chi2(f, x, y, yu)
            par  = np.array(f.values)
            err  = np.array(f.errors)
            valid = True

            fp = FitPar(x  = x,
                        y  = y,
                        yu = yu,
                        f  = f)


        except RuntimeError:
            #warnings.warn(f' fit failed for seed  = {seed} ', UserWarning)
            print(f' fit failed for seed  = {seed}  due to RunTimeError')
            valid = False
            c2 = NN
            par, err = par_and_err_from_seed(seed)

        except RuntimeWarning:
            #warnings.warn(f' fit failed for seed  = {seed} ', UserWarning)
            print(f' fit failed for seed  = {seed}, due to RunTimeWarning, retry fit ')

            fseed = (10*fseed[0], fseed[1], fseed[2] )

            try:

                f     = fitf.fit(fitf.gauss, x, y, fseed, sigma=yu)
                c2    = chi2(f, x, y, yu)
                par  = np.array(f.values)
                err  = np.array(f.errors)
                valid = True

                fp = FitPar(x  = x,
                            y  = y,
                            yu = yu,
                            f  = f)

            except RuntimeWarning:
                print(f' fit failed for seed  = {seed}, due to RunTimeWarning, give up ')
                valid = False
                c2 = NN
                par, err = par_and_err_from_seed(seed)

        except OptimizeWarning:
            #warnings.warn(f' OptimizeWarning was raised for seed  = {seed} ', UserWarning)
            print(f' OptimizeWarning was raised for seed  = {seed} due to OptimizeWarning')
            valid = False
            c2 = NN
            par, err = par_and_err_from_seed(seed)


    fr = FitResult(par = par,
                   err = err,
                   chi2 = c2,
                   valid = valid)

    return fp, fr


def energy_fit(e : np.array,
               nbins   : int,
               range   : Tuple[float],
               n_sigma : float = 3.0)->FitCollection:
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
    bin_size = (range[1] - range[0]) / nbins
    seed = gaussian_parameters(e, range, bin_size)

    fp, fr = gaussian_fit(x, y, seed, n_sigma)

    hp = HistoPar(var      = e,
                  nbins    = nbins,
                  range    = range)

    return FitCollection(fp = fp, hp = hp, fr = fr)


def plot_energy_fit(fc : FitCollection):
    """Takes a KrEvent and a FitPar object and plots fit"""

    if fc.fr.valid:
        par  = fc.fr.par
        err  = fc.fr.err
        _, _, _   = plt.hist(fc.hp.var,
                             bins = fc.hp.nbins,
                             range=fc.hp.range,
                             histtype='step',
                             edgecolor='black',
                             linewidth=1.5,
                             label=r'$\mu={:7.2f} +- {:7.3f},\ \sigma={:7.2f} +- {:7.3f}$'.format(
                               par[1], err[1], par[2], err[2]))

        plt.plot(fc.fp.x, fc.fp.f.fn(fc.fp.x), "r-", lw=4)
    else:
        warnings.warn(f' fit did not succeed, cannot plot ', UserWarning)


def display_energy_fit(fc : FitCollection, figsize : Tuple[int] =(6,6), legend_loc='best'):
    if fc.fr.valid:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        plot_energy_fit(fc)
        ax.legend(fontsize= 10, loc=legend_loc)
    else:
        warnings.warn(f' fit did not succeed, cannot display ', UserWarning)


def print_energy_fit(fc : FitCollection):

    par  = fc.fr.par
    err  = fc.fr.err
    try:
        r  = 2.35 * 100 *  par[2] / par[1]
        fe = np.sqrt(41 / 2458) * r
        print(f'  Fit was valid = {fc.fr.valid}')
        print(f' Emu       = {par[1]} +-{err[1]} ')
        print(f' E sigma   = {par[2]} +-{err[2]} ')
        print(f' chi2    = {fc.fr.chi2} ')

        print(f' sigma E/E (FWHM)     (%) ={r}')
        print(f' sigma E/E (FWHM) Qbb (%) ={fe} ')
    except ZeroDivisionError:
        warnings.warn(f' mu  = {par[1]} ', UserWarning)


def plot_energy_fit_chi2(fc : FitCollection):
    """Takes a KrEvent and a FitPar object and plots fit"""

    if fc.fr.valid:
        x  = fc.fp.x
        f  = fc.fp.f
        y  = fc.fp.y
        yu = fc.fp.yu
        plt.errorbar(x, (f.fn(x) - y) / yu, 1, np.diff(x)[0] / 2, fmt="p", c="k")
        lims = plt.xlim()
        plt.plot(lims, (0, 0), "g--")
        plt.xlim(*lims)
        plt.ylim(-5, +5)
    else:
        warnings.warn(f' fit did not succeed, cannot plot ', UserWarning)


def display_energy_fit_and_chi2(fc : FitCollection, pl : PlotLabels, figsize : Tuple[int] =(6,6),
                                legend_loc : str = 'best'):
    if fc.fr.valid:
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
    else:
        warnings.warn(f' fit did not succeed, cannot display ', UserWarning)


# def energy_fit_XYRange(kre    : KrEvent,
#                        nbins   : int,
#                        range   : Tuple[float],
#                        xr      : Tuple[float],
#                        yr      : Tuple[float],
#                        n_sigma : float = 3.0)->FitCollection:
#
#
#     sel  = in_range(kre.X, *xr) & in_range(kre.Y, *yr)
#     e    = kre.E[sel]
#
#     return energy_fit(e, nbins,erange, n_sigma)


def par_and_err_from_seed(seed : GaussPar) ->Tuple[np.array]:
    par = np.zeros(3)
    err = np.zeros(3)
    par[0] = seed.amp.value
    par[1] = seed.mu.value
    par[2] = seed.std.value
    err[0] = seed.amp.uncertainty
    err[1] = seed.mu.uncertainty
    err[2] = seed.std.uncertainty
    return par, err
