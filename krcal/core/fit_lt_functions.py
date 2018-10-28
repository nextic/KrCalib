"""Module correction_functions.
This module includes the fit functions for the lifetime.

Notes
-----
    KrCalib code depends on the IC library.
    Public functions are documented using numpy style convention

Documentation
-------------
    Insert documentation https
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates  as md
import warnings

from   pandas.core.frame import DataFrame
from typing  import Dict, List, Tuple, Sequence, Iterable, Optional
from numpy.linalg import LinAlgError

from   invisible_cities.core.core_functions import in_range
from   invisible_cities.evm  .ic_containers  import Measurement

from . import fit_functions_ic as fitf
from . fit_functions   import   expo_seed, chi2, chi2f
from . histo_functions import profile1d
from . stat_functions  import  mean_and_std
from . core_functions  import  value_from_measurement
from . core_functions  import  uncertainty_from_measurement
from . core_functions  import  NN

from invisible_cities.core .stat_functions import poisson_sigma
from invisible_cities.icaro. hst_functions import shift_to_bin_centers

from . kr_types import FitPar
from . kr_types import FitParTS
from . kr_types import FitResult
from . kr_types import HistoPar, HistoPar2
from . kr_types import FitCollection, FitCollection2


from . kr_types import FitType, MapType
from . kr_types import Number, Range
from . kr_types import KrEvent
#from . kr_types import TSectorMap, ASectorMap

from scipy.optimize import OptimizeWarning
from . histo_functions import labels
from numpy import sqrt, pi

import sys
import logging
log = logging.getLogger()


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

    Parameters
    ----------
        z
            Array of z values.
        e
            Array of energy values.
        nbins_z
            Number of bins in Z for the profile fit.
        nbins_e
            Number of bins in energy.
        range_z
            Range in Z for fit.
        range_e
            Range in energy.
        fit
            Selects fit type.


    Returns
    -------
        A FitCollection2, which allows passing two fit parameters, histo parameters and fitResult.

    @dataclass
    class FitCollection:
        fp   : FitPar
        hp   : HistoPar
        fr   : FitResult


    @dataclass
    class FitCollection2(FitCollection):
        fp2   : FitPar

    """
    logging.debug(' fit_liftime ')
    logging.debug(f' len (z) ={len(z)}, len (e) ={len(e)} ')
    logging.debug(f' nbins_z ={nbins_z}, nbins_e ={nbins_e} range_z ={range_z} range_e ={range_e} ')

    hp = HistoPar2(var = z,
                   nbins = nbins_z,
                   range = range_z,
                   var2 = e,
                   nbins2 = nbins_e,
                   range2 = range_e)

    if fit == FitType.profile:
        fp, fp2, fr = fit_lifetime_profile(z, e, nbins_z, range_z)
    else:
        fp, fp2, fr = fit_lifetime_unbined(z, e, nbins_z, range_z)

    return FitCollection2(fp = fp, fp2 = fp2, hp = hp, fr = fr)


def fit_lifetime_profile(z : np.array,
                         e : np.array,
                         nbins_z : int,
                         range_z : Tuple[float,float])->Tuple[FitPar, FitPar, FitResult]:
    """
    Make a profile of the input data and fit it to an exponential
    function.
    Parameters
    ----------
        z
            Array of z values.
        e
            Array of energy values.
        nbins_z
            Number of bins in Z for the profile fit.
        range_z
            Range in Z for fit.


    Returns
    -------
        A Tuple with:
            FitPar : Fit parameters (arrays of fitted values and errors, fit function)
            FitPar : Fit parameters (duplicated to make it compatible with fit_liftime_unbined)
            FirResults: Fit results (lt, e0, errors, chi2)

    @dataclass
    class ProfilePar:
        x  : np.array
        y  : np.array
        xu : np.array
        yu : np.array

    @dataclass
    class FitPar(ProfilePar):
        f     : FitFunction

    @dataclass
    class FitResult:
        par  : np.array
        err  : np.array
        chi2 : float
        valid : bool

    """

    logging.debug(' fit_liftime_profile')
    logging.debug(f' len (z) ={len(z)}, len (e) ={len(e)} ')
    logging.debug(f' nbins_z ={nbins_z}, range_z ={range_z} ')
    fp    = None
    valid = True
    c2    = NN
    par   = NN  * np.ones(2)
    err   = NN  * np.ones(2)

    x, y, yu  = profile1d(z, e, nbins_z, range_z)
    xu        = np.diff(x) * 0.5
    seed      = expo_seed(x, y)

    logging.debug(f' after profile: len (x) ={len(x)}, len (y) ={len(y)} ')
    try:
        f      = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
        c2     = f.chi2
        par    = np.array(f.values)
        par[1] = - par[1]
        err    = np.array(f.errors)

        logging.debug(f' e0z ={par[0]} +- {err[0]} ')
        logging.debug(f' lt ={par[1]} +- {err[1]} ')
        logging.debug(f' c2 ={c2} ')
        fp = FitPar(x  = x,
                    y  = y,
                    xu = xu,
                    yu = yu,
                    f  = f.fn)
    except:
        warnings.warn(f' fit failed for seed  = {seed} in fit_lifetime_profile', UserWarning)
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

    Parameters
    ----------
        z
            Array of z values.
        e
            Array of energy values.
        nbins_z
            Number of bins in Z for the profile fit.
        range_z
            Range in Z for fit.


    Returns
    -------
        A Tuple with:
            FitPar : Fit parameters (arrays of fitted values and errors, fit function)
            FitPar : Fit parameters
            FirResults: Fit results (lt, e0, errors, chi2)

    """

    logging.debug(' fit_liftime_unbined')
    logging.debug(f' len (z) ={len(z)}, len (e) ={len(e)} ')
    logging.debug(f' nbins_z ={nbins_z}, range_z ={range_z} ')

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

        logging.debug(f' after profile: len (x) ={len(x)}, len (y) ={len(y)} ')

        c2 = chi2f(lambda z: a * xs + b, 2, xs, ys, yus)

        logging.debug(f' e0z ={par[0]} +- {err[0]} ')
        logging.debug(f' lt ={par[1]} +- {err[1]} ')
        logging.debug(f' c2 ={c2} ')

        fp  = FitPar(x  = x,  y  = y,  xu = xu,  yu = yu,  f  = lambda z: e0 * np.exp(-z/lt))
        fp2 = FitPar(x  = xs, y  = ys, xu = xus, yu = yus, f  = lambda z: a * xs + b)

    except ValueError:
        logging.warn(f'Value Error found in fit_lifetime_unbined: not enough events for fit')
        valid = False

    except TypeError:
        logging.warn(f'Type error found in fit_lifetime_unbined: not enough events for fit')
        valid = False

    except LinAlgError:
        logging.warn(f'LinAlgError error found in fit_lifetime_unbined: not enough events for fit')
        valid = False

    fr = FitResult(par = par,
                   err = err,
                   chi2 = c2,
                   valid = valid)

    return fp, fp2, fr


def time_fcs(ts      : np.array,
             masks   : List[np.array],
             kre     : KrEvent,
             nbins_z : int,
             nbins_e : int,
             range_z : Tuple[float, float],
             range_e : Tuple[float, float],
             energy  : str                 = 'S2e',
             fit     : FitType             = FitType.profile)->FitParTS:
    """
    Fit lifetime of a time series.

    Parameters
    ----------
        ts
            A vector of floats with the (central) values of the time series.
        masks
            A list of boolean vectors specifying the selection masks that define the time series.
        kre
            A kr_event (a subset of dst).
        range_z
            Range in Z for fit.
        nbins_z
            Number of bins in Z for the fit.
        nbins_e
            Number of bins in energy.
        range_z
            Range in Z for fit.
        range_e
            Range in energy.
        energy:
            Takes two values: S2e (uses S2e field in kre) or E (used E field on kre).
            This field allows to select fits over uncorrected (S2e) or corrected (E) energies.
        fit
            Selects fit type.


    Returns
    -------
        A FitParTs:

    @dataclass
    class FitParTS:             # Fit parameters Time Series
        ts   : np.array          # contains the time series (integers expressing time differences)
        e0   : np.array          # e0 fitted in time series
        lt   : np.array
        c2   : np.array
        e0u  : np.array          # e0 error fitted in time series
        ltu  : np.array

    """

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

    logging.debug('function:time_fcs ')
    logging.debug(f' list of kre_event has length {len(kcts)}')
    [logging.debug(f' mask {i} has length {len(mask)}') for i, mask in enumerate(masks)]
    [logging.debug(f' mask {i} has {np.count_nonzero(mask)} True elements')
                   for i, mask in enumerate(masks)]

    if energy == 'S2e':
        #print('S2e')
        fcs =[fit_lifetime(kct.Z, kct.S2e,
                           nbins_z, nbins_e, range_z, range_e, fit) for kct in kcts]
    else:
        #print('E')
        fcs =[fit_lifetime(kct.Z, kct.E,
                           nbins_z, nbins_e, range_z, range_e, fit) for kct in kcts]

    e0s, lts, c2s = pars_from_fcs(fcs)
    #print(value_from_measurement(e0s))
    return FitParTS(ts  = np.array(ts),
                    e0  = value_from_measurement(e0s),
                    lt  = value_from_measurement(lts),
                    c2  = c2s,
                    e0u = uncertainty_from_measurement(e0s),
                    ltu = uncertainty_from_measurement(lts))


def get_time_series(time_bins    : Number,
                    time_range   : Tuple[float, float],
                    kre          : KrEvent)->Tuple[np.array, List[np.array]]:
    """

    Returns a time series (ts) and a list of masks which are used to divide
    the event in time tranches.

        Parameters
        ----------
            time_bins
                Number of time bines.
            time_range
                Time range.
            kre
                A kr_event (a subset of dst).

        Returns
        -------
            A Tuple with:
            np.array       : This is the ts vector
            List[np.array] : This are the list of masks defining the events in the time series.

    """

    logging.debug(f'function: get_time_series')
    nt = time_bins
    x = int((time_range[-1] -  time_range[0]) / nt)
    tfirst = int(time_range[0])
    tlast  = int(time_range[-1])
    if x == 1:
        indx = [(tfirst, tlast)]
    else:
        indx = [(i, i + x) for i in range(tfirst, int(tlast - x), x) ]
        indx.append((x * (nt -1), tlast))

    ts = [(indx[i][0] + indx[i][1]) / 2 for i in range(len(indx))]

    logging.debug(f' number of time bins = {nt}, t_first = {tfirst} t_last = {tlast}')
    logging.debug(f'indx = {indx}')
    logging.debug(f'ts = {ts}')

    masks = [in_range(kre.DT, indx[i][0], indx[i][1]) for i in range(len(indx))]

    return np.array(ts), masks
    

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
def fit_map_rphi(selection_map : Dict[int, List[KrEvent]],
                 event_map     : DataFrame,
                 n_time_bins   : int,
                 time_diffs    : np.array,
                 nbins_z       : int,
                 nbins_e       : int,
                 range_z       : Tuple[float, float],
                 range_e      : Tuple[float, float],
                 energy        : str                 = 'S2e',
                 fit           : FitType             = FitType.profile,
                 n_min         : int                 = 100)->Dict[int, List[FitParTS]]:

    logging.debug(f'function: fit_map_rphi')
    fMAP = {}
    nsectors = len(selection_map.keys())
    for sector in range(nsectors):
        logging.debug(f'Fitting sector {sector}')

        fps = fit_fcs_in_rphi_sectors(sector, selection_map, event_map, n_time_bins, time_diffs,
                                 nbins_z, nbins_e, range_z, range_e, energy, fit, n_min)

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
               energy        : str                 = 'S2e',
               fit           : FitType             = FitType.profile,
               n_min         : int                 = 100)->Dict[int, List[FitParTS]]:

    logging.debug(f'function: fit_map_xy')
    fMAP = {}
    r, c = event_map.shape

    logging.debug(f'event map has {r} bins in x {c} bins in y')
    for i in range(r):
        fMAP[i] = [fit_fcs_in_xy_bin((i,j), selection_map, event_map, n_time_bins, time_diffs,
                                     nbins_z, nbins_e, range_z,range_e, energy, fit, n_min)
                                     for j in range(c) ]
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
    """Returns fits in the bin specified by xybin"""


    i = xybin[0]
    j = xybin[1]
    nevt = event_map[i][j]
    tlast = time_diffs[-1]
    tfrst = time_diffs[0]
    KRE = selection_map
    ts, masks =  get_time_series(n_time_bins, (tfrst, tlast), selection_map[i][j]) # pass one KRE for tsel

    logging.debug(f' --fit_fcs_in_xy_bin called: xy bin = ({i},{j}), with events ={nevt}')

    if nevt > n_min:
        return time_fcs(ts, masks, selection_map[i][j],
                        nbins_z, nbins_e, range_z, range_e, energy, fit)
    else:
        warnings.warn(f'Cannot fit: events in bin[{i}][{j}] ={event_map[i][j]} < {n_min}',
                     UserWarning)

        dum = np.zeros(len(ts), dtype=float)
        dum.fill(np.nan)
        return FitParTS(ts, dum, dum, dum, dum, dum)


def fit_fcs_in_rphi_sectors(sector        : int,
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
    """Returns fits in phi wedges for a given radial sector"""

    wedges    =[len(kre) for kre in selection_map.values() ]  # number of wedges per sector
    tfrst     = time_diffs[0]
    tlast     = time_diffs[-1]

    fps =[]
    for i in range(wedges[sector]):
        if event_map[sector][i] > n_min:
            logging.debug(f'fitting sector/wedge ({sector},{i}) with {event_map[sector][i]} events')
            ts, masks =  get_time_series(n_time_bins, (tfrst, tlast), selection_map[sector][i])

            fp  = time_fcs(ts, masks, selection_map[sector][i],
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

    tfrst     = time_difs[0]
    tlast     = time_difs[-1]
    ts, masks = get_time_series(n_time_bins, (tfrst, tlast), selection_map)

    fp        = time_fcs(masks, kre, nbins_z, nbins_e, range_z, range_e, energy, fit)
    fpf       = time_fcs(masks, kre,  nbins_z, nbins_e, range_zf, range_e, energy, fit)
    fpb       = time_fcs(masks, kre,  nbins_z, nbins_e, range_zb, range_e, energy, fit)

    return fp, fpf, fpb


#experiments
def fit_lifetime_experiments(zs      : np.array,
                             es      : np.array,
                             nbins_z : int      ,
                             nbins_e : int      ,
                             range_z : Tuple[float,float],
                             range_e : Tuple[float,float],
                             fit     : FitType  = FitType.unbined)->List[FitCollection2]:

    return [fit_lifetime(z, e, nbins_z, nbins_e, range_z, range_e, fit) for z,e in zip(zs,es)]


def lt_params_from_fcs(fcs : Iterable[FitCollection])->Iterable[float]:
    e0s   = np.array([fc.fr.par[0] for fc in fcs])
    ue0s  = np.array([fc.fr.err[0] for fc in fcs])
    lts   = np.array([fc.fr.par[1] for fc in fcs])
    ults  = np.array([fc.fr.err[1] for fc in fcs])
    chi2s = np.array([fc.fr.chi2   for fc in fcs])
    return e0s, ue0s, lts, ults, chi2s
