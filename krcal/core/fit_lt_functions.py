import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates  as md
import warnings

from   pandas.core.frame import DataFrame
from typing  import Dict, List, Tuple, Sequence, Iterable, Optional


from   invisible_cities.core.core_functions import in_range
from   invisible_cities.evm  .ic_containers  import Measurement

from . import fit_functions_ic as fitf
from . fit_functions import   expo_seed, chi2, chi2f
from . histo_functions import profile1d
from . stat_functions import  mean_and_std
from . core_functions import  value_from_measurement, uncertainty_from_measurement
from . core_functions import  NN

from invisible_cities.core .stat_functions import poisson_sigma
from invisible_cities.icaro. hst_functions import shift_to_bin_centers


from . kr_types import GaussPar
from . kr_types import FitPar
from . kr_types import FitParTS
from . kr_types import FitResult
from . kr_types import HistoPar, HistoPar2
from . kr_types import FitCollection, FitCollection2
from . kr_types import PlotLabels

from . kr_types import FitType, MapType
from . kr_types import Number, Range
from . kr_types import KrEvent
#from . kr_types import TSectorMap, ASectorMap

from scipy.optimize import OptimizeWarning
from . histo_functions import labels
from numpy import sqrt, pi

import logging
log = logging.getLogger(__name__)



def fit_lifetime(z       : np.array,
                 e       : np.array,
                 nbins_z : int,
                 nbins_e : int,
                 range_z : Tuple[float,float],
                 range_e : Tuple[float,float],
                 fit     : FitType = FitType.unbined)->FitCollection2:
    """
    Fits the lifetime using a profile (FitType.profile) or an unbined
    fit (FitType.unbined).
    """

    hp = HistoPar2(var = z,
                   nbins = nbins_z,
                   range = range_z,
                   var2 = e,
                   nbins2 = nbins_e,
                   range2 = range_e)

    if fit == FitType.profile:
        fp, fp2, fr = fit_lifetime_profile(z, e, nbins_z, range_z)
    else:  # default is unbined
        fp, fp2, fr = fit_lifetime_unbined(z, e, nbins_z, range_z)

    return FitCollection2(fp = fp, fp2 = fp2, hp = hp, fr = fr)


def fit_lifetime_profile(z : np.array,
                         e : np.array,
                         nbins_z : int,
                         range_z : Tuple[float,float])->Tuple[FitPar, FitPar, FitResult]:
    """
    Make a profile of the input data and fit it to an exponential
    function.
    """

    fp    = None
    valid = True
    c2    = NN
    par   = NN  * np.ones(2)
    err   = NN  * np.ones(2)

    x, y, yu  = profile1d(z, e, nbins_z, range_z)
    xu        = np.diff(x) * 0.5
    seed      = expo_seed(x, y)

    try:
        f      = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
        c2     = f.chi2
        par    = np.array(f.values)
        par[1] = - par[1]
        err    = np.array(f.errors)

        fp = FitPar(x  = x,
                    y  = y,
                    xu = xu,
                    yu = yu,
                    f  = f.fn)
    except:
        print(f' fit failed for seed  = {seed} in fit_lifetime_profile')
        valid = False
        raise

    fr = FitResult(par = par,
                   err = err,
                   chi2 = c2,
                   valid = valid)

    return fp, fp, fr


def fit_lifetime_unbined(z       : np.array,
                         e       : np.array,
                         nbins_z : int,
                         range_z : Tuple[float,float])->Tuple[FitPar, FitPar, FitResult]:
    """
    Based on

    numpy.polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False)
    Fit a polynomial p(x) = p[0] * x**deg + ... + p[deg] of degree deg to points (x, y).
    Returns a vector of coefficients p that minimises the squared error.

    E = E0 exp(-z/lt)
    y = -log(E) = (z/lt) - log(E)

    """

    fp    = None
    fp2   = None
    valid = True
    c2    = NN
    par   = NN  * np.ones(2)
    err   = NN  * np.ones(2)
    try:
        el = - np.log(e)
        cc, cov = np.polyfit(z, el, deg=1, full = False, w = None, cov = True )
        a, b = cc[0], cc[1]

        lt   = 1/a
        par[1] = lt
        err[1] = lt**2 * np.sqrt(cov[0, 0])

        e0     = np.exp(-b)
        par[0] = e0
        err[0] = e0    * np.sqrt(cov[1, 1])

        x, y, yu     = profile1d(z, e, nbins_z, range_z)
        xs, ys, yus  = profile1d(z, el, nbins_z, range_z)
        xu           = np.diff(x) * 0.5
        xus          = np.diff(xs) * 0.5

        c2 = chi2f(lambda z: a * xs + b, 2, xs, ys, yus)

        fp  = FitPar(x  = x,  y  = y,  xu = xu,  yu = yu,  f  = lambda z: e0 * np.exp(-z/lt))
        fp2 = FitPar(x  = xs, y  = ys, xu = xus, yu = yus, f  = lambda z: a * xs + b)

    except ValueError:
        print(f'Value Error found in fit_lifetime_unbined')
        valid = False


    fr = FitResult(par = par,
                   err = err,
                   chi2 = c2,
                   valid = valid)

    return fp, fp2, fr


def plot_fit_lifetime(fc : FitCollection):

    if fc.fr.valid:
        par  = fc.fr.par
        err  = fc.fr.err

        if fc.hp:
            plt.hist2d(fc.hp.var,
                        fc.hp.var2,
                        bins = (fc.hp.nbins,fc.hp.nbins2),
                        range= (fc.hp.range,fc.hp.range2))
        x = fc.fp.x
        y = fc.fp.y
        xu = fc.fp.xu
        yu = fc.fp.yu
        f = fc.fp.f

        plt.errorbar(x, y, yu, xu[0], fmt="kp", ms=7, lw=3)
        plt.plot(x, f(x), "r-", lw=4)
        plt.xlabel('Z')
        plt.ylabel('E')
        plt.title(f'Ez0 ={par[0]:7.2f}+-{err[0]:7.3f},   LT={par[1]:7.2f}+-{err[1]:7.3f}')
    else:
        warnings.warn(f' fit did not succeed, cannot plot ', UserWarning)


def plot_fit_lifetime_chi2(fc : FitCollection):

    if fc.fr.valid:
        par  = fc.fr.par
        err  = fc.fr.err
        x = fc.fp2.x
        y = fc.fp2.y
        yu = fc.fp2.yu
        xu = fc.fp2.xu
        f = fc.fp2.f

        lims = (x[0] - np.diff(x)[0] / 2, x[-1] + np.diff(x)[0] / 2)

        plt.errorbar(x, (f(x) - y) / yu, 1, xu[0], fmt="p", c="k")
        plt.plot(lims, (0, 0), "g--")
        plt.xlim(*lims)
        plt.ylim(-5, +5)
        plt.xlabel("Z")

        plt.title(f'chi2')
    else:
        warnings.warn(f' fit did not succeed, cannot plot ', UserWarning)


def print_fit_lifetime(fc : FitCollection):

    if fc.fr.valid:
        par  = fc.fr.par
        err  = fc.fr.err
        print(f' Ez0     = {par[0]} +-{err[0]} ')
        print(f' LT      = {par[1]} +-{err[1]} ')
        print(f' chi2    = {fc.fr.chi2} ')
    else:
        warnings.warn(f' fit did not succeed, cannot print ', UserWarning)


def pars_from_fcs(fcs : List[FitCollection])->Tuple[List[Measurement],
                                                    List[Measurement],
                                                    np.array]:
    E=[]
    LT = []
    C2 = []
    for fc in fcs:
        if fc.fr.valid:
            par  = fc.fr.par
            err  = fc.fr.err
            E.append(Measurement(par[0], err[0]))
            LT.append(Measurement(par[1], err[1]))
            C2.append(fc.fr.chi2)
        else:
            warnings.warn(f' fit did not succeed, returning NaN ', UserWarning)
            E.append(Measurement(NN, NN))
            LT.append(Measurement(NN, NN))
            C2.append(NN)
    return E, LT, np.array(C2)

# Fitting maps
def fit_map(selection_map : Dict[int, List[KrEvent]],
            event_map     : DataFrame,
            n_time_bins   : int,
            time_diffs    : np.array,
            nbins_z       : int,
            nbins_e       : int,
            range_z       : Tuple[float, float],
            range_e       : Tuple[float, float],
            range_chi2    : Tuple[float, float],
            range_lt      : Tuple[float, float],
            energy        : str                 = 'S2e',
            fit           : FitType             = FitType.profile,
            verbose       : bool                = False,
            n_min         : int                 = 100)->Dict[int, List[FitParTS]]:

    fMAP = {}
    nsectors = len(selection_map.keys())
    for sector in range(nsectors):
        logging.debug(f'Fitting sector {sector}')

        fps = fit_fcs_in_sectors(sector, selection_map, event_map, n_time_bins, time_diffs
                                 nbins_z, nbins_e, range_z, range_e, energy, fit, n_min)
        if verbose:
            logging.debug(f' number of wedges fitted in sector {len(fps)}')

        fMAP[sector] = fps

    return fMAP


def fit_map_xy(selection_map : Dict[int, List[KrEvent]],
               event_map     : DataFrame,
               n_time_bins   : int,
               time_diffs     : np.array,
               nbins_z       : int,
               nbins_e       : int,
               range_z       : Tuple[float, float],
               range_e       : Tuple[float, float],
               range_chi2    : Tuple[float, float],
               range_lt      : Tuple[float, float],
               energy        : str                 = 'S2e',
               fit           : FitType             = FitType.profile,
               n_min         : int                 = 100)->Dict[int, List[FitParTS]]:

    fMAP = {}
    r, c = event_map.shape

    for i in range(r):
        FL = []
        for j in range(c):
            nevt = event_map[i][j]
            logging.debug(f'Fitting xy bin = ({i},{j}), with events ={nevt}')

            fps = fit_fcs_in_xy_bin((i,j), selection_map, event_map, n_time_bins, time_diffs,
                                    nbins_z, nbins_e, range_z,range_e, energy, fit, n_min)
        fMAP[i] = FL

    return fMAP


def fit_fcs_in_xy_bin (xybin         : Tuple[int, int],
                       selection_map : Dict[int, List[KrEvent]],
                       event_map     : DataFrame,
                       n_time_bins   : int,
                       time_diffs    : np.array,
                       nbins_z       : int,
                       nbins_e       : int,
                       range_z       : Tuple[float, float],
                       range_e       : Tuple[float, float],
                       energy        : str                 = 'S2e',
                       fit           : FitType             = FitType.profile,
                       n_min         : int                 = 100)->FitParTS:
    """Returns fits in xy bins specified by KRES"""

    i = xybin[0]
    j = xybin[1]
    tlast = time_difs[-1]
    ts, masks =  get_time_series(n_time_bins, tlast, selection_map)

    if event_map[i][j] > n_min:
        logging.debug(f'fitting bin ({i},{j}) with {event_map[i][j]} events')

        return time_fcs(masks, selection_map[i][j],
                        nbins_z, nbins_e, range_z, range_e, energy, fit)
    else:
        warnings.warn(f'Cannot fit: events in bin[{i}][{j}] ={event_map[i][j]} < {n_min}',
                     UserWarning)

        dum = np.zeros(len(ts), dtype=float)
        dum.fill(np.nan)
        return FitParTS(ts, dum, dum, dum, dum, dum)


def fit_fcs_in_sectors(sector        : int,
                       selection_map : Dict[int, List[KrEvent]],
                       event_map     : DataFrame,
                       n_time_bins   : int,
                       time_diffs    : np.array,
                       nbins_z       : int,
                       nbins_e       : int,
                       range_z       : Tuple[float, float],
                       range_e       : Tuple[float, float],
                       energy        : str                 = 'S2e',
                       fit           : FitType             = FitType.profile,
                       n_min         : int                 = 100)->List[FitParTS]:
    """Returns fits in Rphi sectors specified by KRES"""

    wedges    =[len(kre) for kre in selection_map.values() ]  # number of wedges per sector
    tlast     = time_difs[-1]
    ts, masks =  get_time_series(n_time_bins, tlast, selection_map)

    fps =[]
    for i in range(wedges[sector]):
        if event_map[sector][i] > n_min:
            logging.debug(f'fitting sector/wedge ({sector},{i}) with {event_map[sector][i]} events')

            fp  = time_fcs(masks, selection_map[sector][i],
                            nbins_z, nbins_e, range_z, range_e, energy, fit)
        else:
            warnings.warn(f'Cannot fit: events in s/w[{sector}][{i}] ={event_map[sector][i]} < {n_min}',
                         UserWarning)

            dum = np.zeros(len(ts), dtype=float)
            dum.fill(np.nan)
            fp  = FitParTS(ts, dum, dum, dum, dum, dum)

        fps.append(fp)
    return fps


def fb_fits(n_time_bins : int,
            time_diffs  : np.array,
            kre         : KrEvent,
            nbins_z     : int,
            nbins_e     : int,
            range_z     : Tuple[float, float] = (50,550),
            range_zf    : Tuple[float, float] = (50,300),
            range_zb    : Tuple[float, float] = (300,550),
            range_e     : Tuple[float, float] = (7000, 12000),
            energy      : str                 = 'S2e',
            fit         : FitType             = FitType.profile)->Iterable[FitParTS]:
    """Returns fits to full/forward/backward chamber"""

    tlast     = time_difs[-1]
    ts, masks = get_time_series(n_time_bins, tlast, selection_map)

    fp        = time_fcs(masks, kre, nbins_z, nbins_e, range_z, range_e, energy, fit)
    fpf       = time_fcs(masks, kre,  nbins_z, nbins_e, range_zf, range_e, energy, fit)
    fpb       = time_fcs(masks, kre,  nbins_z, nbins_e, range_zb, range_e, energy, fit)

    return fp, fpf, fpb


def time_fcs(masks   : List[np.array],
             kre     : KrEvent,
             nbins_z : int,
             nbins_e : int,
             range_z : Tuple[float, float] = (100,550),
             range_e : Tuple[float, float] = (8000, 12000),
             energy  : str                 = 'S2e',
             fit     : FitType             = FitType.profile)->FitParTS:
    """Fit lifetime of a time series. """

    kcts = [KrEvent(X   = kre.X[sel_mask],
                    Y   = kre.Y[sel_mask],
                    Z   = kre.Z[sel_mask],
                    R   = kre.R[sel_mask],
                    Phi = kre.Phi[sel_mask],
                    T   = kre.T[sel_mask],
                    DT  = kre.DT[sel_mask],
                    S2e = kre.S2e[sel_mask],
                    S1e = kre.S1e[sel_mask],
                    S2q = kre.S2q[sel_mask],
                    E   = kre.E[sel_mask],
                    Q   = kre.Q[sel_mask]) for sel_mask in masks]

    if energy == 'S2e':
        fcs =[fit_lifetime(kct.Z, kct.S2e, fit,
                           nbins_z, nbins_e, range_z, range_e) for kct in kcts]
    else:
        fcs =[fit_lifetime(kct.Z, kct.E, fit,
                           nbins_z, nbins_e, range_z, range_e) for kct in kcts]

    e0s, lts, c2s = pars_from_fcs(fcs)
    return FitParTS(ts  = np.array(ts),
                    e0  = value_from_measurement(e0s),
                    lt  = value_from_measurement(lts),
                    c2  = c2s,
                    e0u = uncertainty_from_measurement(e0s),
                    ltu = uncertainty_from_measurement(lts))


def get_time_series(nt    : int,
                    tlast : int,
                    KRES  : KrEvent)->Tuple[List[float], List[np.array]]:

    x = int(tlast / nt)
    if x == 1:
        indx = [(0, int(tlast))]
    else:
        indx = [(i, i + nt) for i in range(0, int(tlast - nt), nt) ]
        indx.append((x * nt, int(tlast)))

    ts = [(indx[i][0] + indx[i][1]) / 2 for i in range(len(indx))]

    logging.debug(f' number of time bins = {nt}, t_last = {tlast}')
    logging.debug(f'indx = {indx}')
    logging.debug(f'ts = {ts}')

    masks = [in_range(KRES.DT, indx[i][0], indx[i][1]) for i in range(len(indx))]

    return ts, masks

#experiments 
def fit_lifetime_experiments(zs      : np.array,
                             es      : np.array,
                             nbins_z : int      ,
                             nbins_e : int      ,
                             range_z : Range    ,
                             range_e : Range    ,
                             fit     : FitType  = FitType.unbined)->List[FitCollection2]:

    return [fit_lifetime(z, e, nbins_z, nbins_e, range_z, range_e, fit) for z,e in zip(zs,es)]


def lt_params_from_fcs(fcs : Iterable[FitCollection])->Iterable[float]:
    e0s   = np.array([fc.fr.par[0] for fc in fcs])
    ue0s  = np.array([fc.fr.err[0] for fc in fcs])
    lts   = np.array([fc.fr.par[1] for fc in fcs])
    ults  = np.array([fc.fr.err[1] for fc in fcs])
    chi2s = np.array([fc.fr.chi2   for fc in fcs])
    return e0s, ue0s, lts, ults, chi2s


def plot_fit_lifetime_and_chi2(fc : FitCollection, figsize=(10,10)):
    fig = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(1, 2, 1)
    plot_fit_lifetime(fc)

    ax      = fig.add_subplot(1, 2, 2)
    plot_fit_lifetime_chi2(fc)

    plt.tight_layout()
