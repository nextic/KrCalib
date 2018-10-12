import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing      import List, Dict, Tuple, Sequence, Iterable
import warnings

#import matplotlib.dates  as md
from   invisible_cities.core.core_functions import in_range
#import invisible_cities.core.fit_functions as fitf
from . import fit_functions_ic as fitf

from   invisible_cities.evm  .ic_containers  import Measurement
from . fit_functions import chi2
from . stat_functions import mean_and_std

from pandas import DataFrame
from . kr_types import Number, Range

from invisible_cities.core .stat_functions import poisson_sigma
from invisible_cities.icaro. hst_functions import shift_to_bin_centers
from invisible_cities.types.ic_types       import NN

from . kr_types import GaussPar
from . kr_types import FitPar
from . kr_types import FitResult
from . kr_types import HistoPar
from . kr_types import FitCollection
from . kr_types import PlotLabels
from . kr_types import KrEvent

from . kr_types        import PlotLabels
from . histo_functions import plot_histo
from scipy.optimize import OptimizeWarning
from numpy import sqrt, pi
from . stat_functions       import relative_error_ratio


def gfit(x     : np.array,
         y     : np.array,
         yu    : np.array,
         fseed : Tuple[float, float, float]) ->Tuple[FitPar, FitResult]:

    f     = fitf.fit(fitf.gauss, x, y, fseed, sigma=yu)
    c2    = chi2(f, x, y, yu)
    par  = np.array(f.values)
    err  = np.array(f.errors)
    xu   = np.diff(x) * 0.5

    fr = FitResult(par = par,
                   err = err,
                   chi2 = c2,
                   valid = True)
    fp = FitPar(x  = x, y  = y, xu = xu, yu = yu, f  = f.fn)

    return fp, fr

def gaussian_parameters(x : np.array, range : Tuple[Number], bin_size : float = 1)->GaussPar:
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
                 n_sigma : int)  ->Tuple[FitPar, FitResult]:
    """Gaussian fit to x,y variables, with fit range defined by n_sigma"""

    mu  = seed.mu.value
    std = seed.std.value
    amp = seed.amp.value
    fit_range = mu - n_sigma * std, mu + n_sigma * std

    x, y      = x[in_range(x, *fit_range)], y[in_range(x, *fit_range)]
    yu        = poisson_sigma(y)
    fseed     = (amp, mu, std)

    par, err = par_and_err_from_seed(seed)
    fr = FitResult(par = par,
                   err = err,
                   chi2 = NN,
                   valid = False)
    fp = None

    with warnings.catch_warnings():
        warnings.filterwarnings('error')  # in order to handle fit failures here
        try:
            fp, fr = gfit(x, y, yu, fseed)
        except RuntimeWarning:   # this is the most usual failure, and usually solved trying fitx
                                 # with a different seed
            print(f' fit failed for seed  = {seed}, due to RunTimeWarning, retry fit ')
            fseed = (10*fseed[0], fseed[1], fseed[2] )
            try:
                fp, fr = gfit(x, y, yu, fseed)
            except RuntimeWarning: #  Give up on second failure
                print(f' fit failed for seed  = {seed}, due to RunTimeWarning, give up ')
        except OptimizeWarning:
            print(f' OptimizeWarning was raised for seed  = {seed} due to OptimizeWarning')
        except RuntimeError:
            print(f' fit failed for seed  = {seed}  due to RunTimeError')
        except TypeError:
            print(f' fit failed for seed  = {seed}  due to TypeError')

    return fp, fr


def fit_energy(e : np.array,
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


def plot_fit_energy(fc : FitCollection):

    if fc.fr.valid:
        par  = fc.fr.par
        x    = fc.hp.var
        r    = 2.35 * 100 *  par[2] / par[1]
        entries  =  f'Entries = {len(x)}'
        mean     =  r'$\mu$ = {:7.2f}'.format(par[1])
        sigma    =  r'$\sigma$ = {:7.2f}'.format(par[2])
        rx       =  r'$\sigma/mu$ (FWHM)  = {:7.2f}'.format(r)
        stat     =  f'{entries}\n{mean}\n{sigma}\n{rx}'

        _, _, _   = plt.hist(fc.hp.var,
                             bins = fc.hp.nbins,
                             range=fc.hp.range,
                             histtype='step',
                             edgecolor='black',
                             linewidth=1.5,
                             label=stat)

        plt.plot(fc.fp.x, fc.fp.f(fc.fp.x), "r-", lw=4)
    else:
        warnings.warn(f' fit did not succeed, cannot plot ', UserWarning)


def print_fit_energy(fc : FitCollection):

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


def plot_fit_energy_chi2(fc : FitCollection):

    if fc.fr.valid:
        x  = fc.fp.x
        f  = fc.fp.f
        y  = fc.fp.y
        yu = fc.fp.yu
        plt.errorbar(x, (f(x) - y) / yu, 1, np.diff(x)[0] / 2, fmt="p", c="k")
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



def resolution_selected_r_z(Rr : Tuple[float, float],
                            Zr : Tuple[float, float],
                            R  : np.array,
                            Z  : np.array,
                            E  : np.array,
                            enbins = 25,
                            erange = (10e+3, 12500))->FitCollection:


    sel_r = in_range(R, *Rr)
    sel_z = in_range(Z, *Zr)
    sel   = sel_r & sel_z
    fc = fit_energy(E[sel], nbins=enbins, range=erange)
    return fc


def resolution_r_z(Ri : Iterable[float],
                   Zi : Iterable[float],
                   R : np.array,
                   Z : np.array,
                   E : np.array,
                   enbins = 25,
                   erange = (10e+3, 12500),
                   ixy = (3,4),
                   fdraw = True,
                   fprint = True,
                   figsize = (14,10))->Tuple[DataFrame, DataFrame]:
    if fdraw:
        fig       = plt.figure(figsize=figsize)
    FC = {}
    FCE = {}
    FCE = {}
    j=0
    ix = ixy[0]
    iy = ixy[1]
    for i, r in enumerate(Ri):
        ZR = []
        ZRE = []
        for z in Zi:
            Rr = 0, r
            Zr = 0, z
            j+=1
            if fdraw:
                ax  = fig.add_subplot(ix, iy, j)
            fc = resolution_selected_r_z(Rr, Zr, R, Z, E, enbins, erange)
            if fdraw:
                plot_fit_energy(fc)
                plot_histo(PlotLabels('E','Entries',f' 0 < R < {r} 0 < z < {z}'), ax, legend= True,
                        legendsize=10, legendloc='best', labelsize=11)
            if fprint:
                print(f'0 < R < {r} 0 < z < {z}')
                print_fit_energy(fc)

            par     = fc.fr.par
            err     = fc.fr.err
            fwhm    = 2.35 * 100 *  par[2] / par[1]
            a       = 2.35 * 100 *  par[2]
            b       = par[1]
            sigma_a = 2.35 * 100 * err[2]
            sigma_b = err[1]

            rer = relative_error_ratio(a, sigma_a, b, sigma_b)
            ZR.append(fwhm)
            ZRE.append(rer*fwhm)
        FC[i] = ZR
        FCE[i] = ZRE

    plt.tight_layout()
    return pd.DataFrame.from_dict(FC), pd.DataFrame.from_dict(FCE)



def plot_resolution_r_z(Ri : Iterable[float],
                        Zi : Iterable[float],
                        FC : DataFrame,
                        FCE : DataFrame,
                        figsize = (14,10)):

    def extrapolate_to_qbb(es : float)->float:
        return np.sqrt(41 / 2458) * es



    fig       = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1)
    ax2 = ax.twinx()
    Zcenters =np.array(list(Zi))
    for i in FC.columns:
        label = f'0 < R < {Ri[i]:2.0f}'

        es = FC[i].values
        eus = FCE[i].values
        qes = extrapolate_to_qbb(es)
        qeus = extrapolate_to_qbb(eus)
        ax.errorbar(Zcenters, es, eus,
                    label = label,
                    fmt='o', markersize=10., elinewidth=10.)
        ax2.errorbar(Zcenters, qes, qeus,
                    label = label,
                    fmt='o', markersize=10., elinewidth=10.)
    plt.grid(True)
    ax.set_xlabel(' z (mm)')
    ax.set_ylabel('resolution FWHM (%)')
    ax2.set_ylabel('resolution Qbb FWHM (%)')

    plt.legend()
    plt.show()


def fit_gaussian_experiments(exps    : np.array,
                             nbins   : int       = 50,
                             range   : Range     = (9e+3, 11e+3),
                             n_sigma : int       =3)->List[FitCollection]:
    return [fit_energy(e, nbins, range, n_sigma) for e in exps]


def fit_gaussian_experiments_variable_mean_and_std(means   : np.array,
                                                   stds    : np.array,
                                                   exps    : np.array,
                                                   bins    : int = 50,
                                                   n_sigma : int =3)->Iterable[List[float]]:
    l = len(stds)
    SEED = []
    MU = []
    STD = []
    AVG = []
    RMS = []
    CHI2 = []
    for i,mean in enumerate(means):
        for j, std in enumerate(stds):
            k = i*l + j
            e = exps[k]
            r = mean - n_sigma * std, mean + n_sigma * std
            bin_size = (r[1] - r[0]) / bins
            gp = gaussian_parameters(e, range = r, bin_size=bin_size)
            fc = fit_energy(e, nbins=bins, range=r, n_sigma = n_sigma)
            SEED.append(Measurement(mean, std))
            MU.append(Measurement(fc.fr.par[1], fc.fr.err[1]))
            STD.append(Measurement(fc.fr.par[2], fc.fr.err[2] ))
            AVG.append(gp.mu)
            RMS.append(gp.std)
            CHI2.append(fc.fr.chi2)

    return SEED, MU, STD, AVG, RMS, CHI2


def gaussian_params_from_fcs(fcs : FitCollection) ->Iterable[float]:
    mus   = np.array([fc.fr.par[1] for fc in fcs])
    umus  = np.array([fc.fr.err[1] for fc in fcs])
    stds  = np.array([fc.fr.par[2] for fc in fcs])
    ustds = np.array([fc.fr.err[2] for fc in fcs])
    chi2s = np.array([fc.fr.chi2   for fc in fcs])
    return mus, umus, stds, ustds, chi2s


def par_and_err_from_seed(seed : GaussPar) ->Tuple[np.array, np.array]:
    par = np.zeros(3)
    err = np.zeros(3)
    par[0] = seed.amp.value
    par[1] = seed.mu.value
    par[2] = seed.std.value
    err[0] = seed.amp.uncertainty
    err[1] = seed.mu.uncertainty
    err[2] = seed.std.uncertainty
    return par, err
