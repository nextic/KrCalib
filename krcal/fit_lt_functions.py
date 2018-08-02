import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates  as md
from typing      import List, Tuple
import warnings

from   invisible_cities.core.core_functions import in_range
from   invisible_cities.evm  .ic_containers  import Measurement

from . import fit_functions_ic as fitf
from . fit_functions import chi2
from . core_functions import mean_and_std
from . core_functions import Number

from invisible_cities.core .stat_functions import poisson_sigma
from invisible_cities.icaro. hst_functions import shift_to_bin_centers
from invisible_cities.types.ic_types       import NN

from . kr_types import GaussPar
from . kr_types import FitPar
from . kr_types import FitResult
from . kr_types import HistoPar, HistoPar2
from . kr_types import FitCollection
from . kr_types import PlotLabels

from . histo_functions import labels
from scipy.optimize import OptimizeWarning
from numpy import sqrt, pi


from . fit_functions import expo_seed
from . fit_functions import to_relative
from . fit_functions import fit_profile_1d_expo
from . kr_types import FitPar
from . kr_types import FitResult
from . kr_types import HistoPar
from . kr_types import FitCollection
from . kr_types import PlotLabels

from . histo_functions import labels
from numpy import sqrt, pi


def fit_lifetime(z : np.array,
                 e : np.array,
                 nbins_z : int,
                 nbins_e : int,
                 range_z : Tuple[float],
                 range_e : Tuple[float])->FitCollection:
    """
    Make a profile of the input data and fit it to an exponential
    function with the parameters automatically estimated.
    """
    x, y, yu     = fitf.profileX(z, e, nbins_z)
    valid_points = yu > 0

    x    = x [valid_points]
    y    = y [valid_points]
    yu   = yu[valid_points]

    seed = expo_seed(x, y)

    f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)

    c2    = chi2(f, x, y, yu)
    par  = np.array(f.values)
    err  = np.array(f.errors)
    valid = True

    fp = FitPar(x  = x,
                y  = y,
                yu = yu,
                f  = f)

    fr = FitResult(par = par,
                   err = err,
                   chi2 = c2,
                   valid = valid)

    hp = HistoPar2(var = z,
                  nbins = nbins_z,
                  range = range_z,
                  var2 = e,
                  nbins2 = nbins_e,
                  range2 = range_e)

    return FitCollection(fp = fp, hp = hp, fr = fr)


def plot_fit_lifetime(fc : FitCollection):

    if fc.fr.valid:
        par  = fc.fr.par
        err  = fc.fr.err
        plt.hist2d(fc.hp.var,
                   fc.hp.var2,
                   bins = (fc.hp.nbins,fc.hp.nbins2),
                   range= (fc.hp.range,fc.hp.range2))

        plt.errorbar(fc.fp.x, fc.fp.y, fc.fp.yu, np.diff(fc.fp.x)[0]/2, fmt="kp", ms=7, lw=3)
        plt.plot(fc.fp.x, fc.fp.f.fn(fc.fp.x), "r-", lw=4)
        plt.xlabel('Z')
        plt.ylabel('E')
        plt.title(f'Ez0 ={par[0]:7.2f}+-{err[0]:7.3f},   LT={par[1]:7.2f}+-{err[1]:7.3f}')
    else:
        warnings.warn(f' fit did not succeed, cannot plot ', UserWarning)

def display_fit_lifetime(fc : FitCollection, figsize : Tuple[int] =(6,6)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        plot_fit_lifetime(fc)


def display_fit_lifetime_and_chi2(fc : FitCollection, figsize : Tuple[int] =(6,6)):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        # create a figure and plot 2D histogram and profile
        frame_data = plt.gcf().add_axes((.1, .3, .8, .6))
        plot_fit_lifetime(fc)

        #    add a second frame
        #
        lims = plt.xlim()
        frame_res = plt.gcf().add_axes((.1, .1, .8, .2))
        # Plot (y - f(x)) / sigma(y) as a function of x
        plt.errorbar(fc.fp.x, (fc.fp.f.fn(fc.fp.x) - fc.fp.y) / fc.fp.yu, 1, np.diff(fc.fp.x)[0] / 2,
                         fmt="p", c="k")
        plt.plot(lims, (0, 0), "g--")
        plt.xlim(*lims)
        plt.ylim(-5, +5)
        plt.xlabel("Z")




# def fit_lifetime_slices(kre : KrEvent,
#                         krnb: KrNBins,
#                         krb : KrBins,
#                         krr : KrRanges,
#                         fit_var = "E",
#                         min_entries = 1e2)->KrLTSlices:
#
#     """
#     Slice the data in x and y, make the profile in z of E,
#     fit it to a exponential and return the relevant values.
#     """
#
#     xybins   = krb.XY
#     nbins_xy = np.size (xybins) - 1
#     nbins_z = krnb.Z
#     nbins   = nbins_xy, nbins_xy
#     const   = np.zeros(nbins)
#     slope   = np.zeros(nbins)
#     constu  = np.zeros(nbins)
#     slopeu  = np.zeros(nbins)
#     chi2    = np.zeros(nbins)
#     valid   = np.zeros(nbins, dtype=bool)
#     zrange = krr.Z
#
#     for i in range(nbins_xy):
#         sel_x = in_range(kre.X, *xybins[i:i + 2])
#         for j in range(nbins_xy):
#             #print(f' bin =({i},{j});  index = {index}')
#             sel_y = in_range(kre.Y, *xybins[j:j + 2])
#             sel   = sel_x & sel_y
#             entries = np.count_nonzero(sel)
#             if entries < min_entries:
#                 #print(f'entries ={entries} not enough  to fit bin (i,j) =({i},{j})')
#                 valid [i, j] = False
#                 continue
#
#             try:
#                 z = kre.Z[sel]
#                 t = kre.E[sel]
#                 if fit_var == "Q":
#                     t = kre.Q[sel]
#
#                 x, y, yu = fitf.profileX(z, t, nbins_z, zrange)
#
#                 seed = expo_seed(x, y)
#                 f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
#                 re = np.abs(f.errors[1] / f.values[1])
#                 #print(f' E +- Eu = {f.values[0]} +- {f.errors[0]}')
#                 #print(f' LT +- LTu = {-f.values[1]} +- {f.errors[1]}')
#                 #print(f' LTu/LT = {re} chi2 = {f.chi2}')
#
#                 const [i, j] = f.values[0]
#                 constu[i, j] = f.errors[0]
#                 slope [i, j] = -f.values[1]
#                 slopeu[i, j] = f.errors[1]
#                 chi2  [i, j] = f.chi2
#                 valid [i, j] = True
#
#                 if re > 0.5:
#                     # print(f'Relative error to large, re ={re} for bin (i,j) =({i},{j})')
#                     # print(f' LT +- LTu = {-f.values[1]} +- {f.errors[1]}')
#                     # print(f' LTu/LT = {re} chi2 = {f.chi2}')
#                     valid [i, j] = False
#
#             except:
#                 print(f'fit failed for bin (i,j) =({i},{j})')
#                 pass
#
#     return KrLTSlices(Es  = Measurement(const, constu),
#                        LT   = Measurement(slope, slopeu),
#                        chi2 = chi2,
#                        valid = valid)


# def lifetimes_in_TRange(kre : KrEvent,
#                         krnb: KrNBins,
#                         krb : KrBins,
#                         krr : KrRanges,
#                         TL)->List[KrFit]:
#     """ Plots lifetime fitted to a range of T values"""
#
#     # Specify the range and number of bins in Z
#     Znbins = krnb.Z
#     Zrange = krr.Z
#
#     kfs=[]
#     for  tlim in TL:
#
#         # select data
#         kre_t = select_in_TRange(kre, *tlim)
#         z, e = kre_t.Z, kre_t.E
#
#         x, y, yu = fitf.profileX(z, e, Znbins, Zrange)
#         # Fit profile to an exponential
#         seed = expo_seed(x, y)
#         f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
#
#         kf = KrFit(par  = np.array(f.values),
#                    err  = np.array(f.errors),
#                    chi2 = chi2(f, x, y, yu))
#
#         #krf.print_fit(kf)
#         kfs.append(kf)
#
#     return kfs
#
#
# def s12_time_profile(krdst, Tnbins, Trange, timeStamps,
#                      s2lim=(8e+3, 1e+4), s1lim=(10,11), figsize=(8,8)):
#
#     xfmt = md.DateFormatter('%d-%m %H:%M')
#     fig = plt.figure(figsize=figsize)
#
#     x, y, yu = fitf.profileX(krdst.T, krdst.E, Tnbins, Trange)
#     ax = fig.add_subplot(1, 2, 1)
#     #plt.figure()
#     #ax=plt.gca()
#     #fig.add_subplot(1, 2, 1)
#     ax.xaxis.set_major_formatter(xfmt)
#     plt.errorbar(timeStamps, y, yu, fmt="kp", ms=7, lw=3)
#     plt.xlabel('date')
#     plt.ylabel('S2 (pes)')
#     plt.ylim(s2lim)
#     plt.xticks( rotation=25 )
#
#     x, y, yu = fitf.profileX(krdst.T, krdst.S1, Tnbins, Trange)
#     ax = fig.add_subplot(1, 2, 2)
#     #ax=plt.gca()
#
#     #xfmt = md.DateFormatter('%d-%m %H:%M')
#     ax.xaxis.set_major_formatter(xfmt)
#     plt.errorbar(timeStamps, y, yu, fmt="kp", ms=7, lw=3)
#     plt.xlabel('date')
#     plt.ylabel('S1 (pes)')
#     plt.ylim(s1lim)
#     plt.xticks( rotation=25 )
#     plt.tight_layout()
#
#
# def select_in_XYRange(kre : KrEvent, xyr : XYRanges)->KrEvent:
#     """ Selects a KrEvent in  a range of XY values"""
#     xr = xyr.X
#     yr = xyr.Y
#     sel  = in_range(kre.X, *xr) & in_range(kre.Y, *yr)
#
#     return KrEvent(X = kre.X[sel],
#                    Y = kre.Y[sel],
#                    Z = kre.Z[sel],
#                    E = kre.E[sel],
#                    S1 = kre.S1[sel],
#                    T = kre.T[sel],
#                    Q = kre.Q[sel])
#
#
# def select_in_TRange(kre : KrEvent, tmin : float, tmax : float)->KrEvent:
#     """ Selects a KrEvent in  a range of T values"""
#
#     sel  = in_range(kre.T, tmin, tmax)
#
#     return KrEvent(X = kre.X[sel],
#                    Y = kre.Y[sel],
#                    Z = kre.Z[sel],
#                    E = kre.E[sel],
#                    S1 = kre.S1[sel],
#                    T = kre.T[sel],
#                    Q = kre.Q[sel])
#
#
# def lifetime_in_XYRange(kre : KrEvent,
#                         krnb: KrNBins,
#                         krb : KrBins,
#                         krr : KrRanges,
#                         xyr : XYRanges)->KrFit:
#     """ Fits lifetime to a range of XY values"""
#
#     # select data in region defined by xyr
#     kre_xy = select_in_XYRange(kre, xyr)
#     z, e = kre_xy.Z, kre_xy.E
#
#     # Specify the range and number of bins in Z
#     Znbins = krnb.Z
#     Zrange = krr.Z
#
#     # create a figure and plot 2D histogram and profile
#     frame_data = plt.gcf().add_axes((.1, .3, .8, .6))
#     plt.hist2d(z, e, (krb.Z, krb.E))
#     x, y, yu = fitf.profileX(z, e, Znbins, Zrange)
#     plt.errorbar(x, y, yu, np.diff(x)[0]/2, fmt="kp", ms=7, lw=3)
#
#     # Fit profile to an exponential
#     seed = expo_seed(x, y)
#     f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
#
#     # plot fitted value
#     plt.plot(x, f.fn(x), "r-", lw=4)
#
#     # labels and ticks
#     frame_data.set_xticklabels([])
#     labels("", "Energy (pes)", "Lifetime fit")
#
#     # add a second frame
#
#     lims = plt.xlim()
#     frame_res = plt.gcf().add_axes((.1, .1, .8, .2))
#     # Plot (y - f(x)) / sigma(y) as a function of x
#     plt.errorbar(x, (f.fn(x) - y) / yu, 1, np.diff(x)[0] / 2,
#                  fmt="p", c="k")
#     plt.plot(lims, (0, 0), "g--")
#     plt.xlim(*lims)
#     plt.ylim(-5, +5)
#     labels("Drift time (µs)", "Standarized residual")
#
#     return KrFit(par  = np.array(f.values),
#                  err  = np.array(f.errors),
#                  chi2 = chi2(f, x, y, yu))
#
#
# def lifetimes_in_XYRange(kre : KrEvent,
#                         krnb: KrNBins,
#                         krb : KrBins,
#                         krr : KrRanges,
#                         xyr : XYRanges,
#                         XL = [(-125, -75), (-125, -75), (75, 125),(75, 125)],
#                         YL = [(-125, -75), (75, 125), (75, 125),(-125, -75)],
#                         nx=2, ny=2,
#                         figsize=(8,8))->KrFit:
#     """ Plots lifetime fitted to a range of XY values"""
#
#
#     # Specify the range and number of bins in Z
#     Znbins = krnb.Z
#     Zrange = krr.Z
#
#     fig = plt.figure(figsize=figsize)
#
#     # XL = [(-125, -75), (-125, -75), (75, 125),(75, 125)]
#     # YL = [(-125, -75), (75, 125), (75, 125),(-125, -75)]
#     KF =[]
#     for i, pair in enumerate(zip(XL,YL)):
#         xlim = pair[0]
#         ylim = pair[1]
#         print(f'xlim = {xlim}, ylim ={ylim}')
#
#         # select data in region defined by xyr
#         xyr = XYRanges(X=xlim, Y=ylim )
#         kre_xy = select_in_XYRange(kre, xyr)
#         z, e = kre_xy.Z, kre_xy.E
#
#         ax = fig.add_subplot(nx, ny, i+1)
#         x, y, yu = fitf.profileX(z, e, Znbins, Zrange)
#         plt.errorbar(x, y, yu, np.diff(x)[0]/2, fmt="kp", ms=7, lw=3)
#
#         # Fit profile to an exponential
#         seed = expo_seed(x, y)
#         f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
#
#         # plot fitted value
#         plt.plot(x, f.fn(x), "r-", lw=4)
#
#
#         labels("", "Energy (pes)", "Lifetime fit")
#
#
#         kf = KrFit(par  = np.array(f.values),
#                    err  = np.array(f.errors),
#                    chi2 = chi2(f, x, y, yu))
#         KF.append(kf)
#     return KF
#
#
#
#
# def fit_slices_2d_expo(kre : KrEvent,
#                        krnb: KrNBins,
#                        krb : KrBins,
#                        krr : KrRanges,
#                        fit_var = "E",
#                        min_entries = 1e2)->KrLTSlices:
#
#     """
#     Slice the data in x and y, make the profile in z of E,
#     fit it to a exponential and return the relevant values.
#     """
#
#     xbins   = krb.XY
#     ybins   = krb.XY
#     nbins_x = np.size (xbins) - 1
#     nbins_y = np.size (ybins) - 1
#     nbins_z = krnb.Z
#     nbins   = nbins_x, nbins_y
#     const   = np.zeros(nbins)
#     slope   = np.zeros(nbins)
#     constu  = np.zeros(nbins)
#     slopeu  = np.zeros(nbins)
#     chi2    = np.zeros(nbins)
#     valid   = np.zeros(nbins, dtype=bool)
#     zrange = krr.Z
#
#     for i in range(nbins_x):
#         sel_x = in_range(kre.X, *xbins[i:i + 2])
#         for j in range(nbins_y):
#             sel_y = in_range(kre.Y, *ybins[j:j + 2])
#             sel   = sel_x & sel_y
#             if np.count_nonzero(sel) < min_entries:
#                 print(f'entries ={entries} not enough  to fit bin (i,j) =({i},{j})')
#                 valid [i, j] = False
#                 continue
#
#             try:
#                 z = kre.Z[sel]
#                 t = kre.E[sel]
#                 if fit_var == "Q":
#                     t = kre.Q[sel]
#
#                 f = fit_profile_1d_expo(z, t, nbins_z, xrange=zrange)
#                 re = np.abs(f.errors[1] / f.values[1])
#
#                 if re > 0.5:
#                     print(f'Relative error to large, re ={re} for bin (i,j) =({i},{j})')
#                     valid [i, j] = False
#
#                 const [i, j] = f.values[0]
#                 constu[i, j] = f.errors[0]
#                 slope [i, j] = -f.values[1]
#                 slopeu[i, j] = f.errors[1]
#                 chi2  [i, j] = f.chi2
#                 valid [i, j] = True
#             except:
#                 pass
#     return KrLTSlices(Ez0  = Measurement(const, constu),
#                        LT   = Measurement(slope, slopeu),
#                        chi2 = chi2,
#                        valid = valid)
#
#
# def fit_and_plot_slices_2d_expo(kre : KrEvent,
#                                 krnb: KrNBins,
#                                 krb : KrBins,
#                                 krr : KrRanges,
#                                 fit_var = "E",
#                                 min_entries = 1e2,
#                                 figsize=(12,12))->KrLTSlices:
#
#     """
#     Slice the data in x and y, make the profile in z of E,
#     fit it to a exponential and return the relevant values.
#     """
#
#     xybins   = krb.XY
#     nbins_xy = np.size (xybins) - 1
#     nbins_z = krnb.Z
#     nbins   = nbins_xy, nbins_xy
#     const   = np.zeros(nbins)
#     slope   = np.zeros(nbins)
#     constu  = np.zeros(nbins)
#     slopeu  = np.zeros(nbins)
#     chi2    = np.zeros(nbins)
#     valid   = np.zeros(nbins, dtype=bool)
#     zrange = krr.Z
#
#     fig = plt.figure(figsize=figsize) # Creates a new figure
#     k=0
#     index = 0
#     for i in range(nbins_xy):
#         sel_x = in_range(kre.X, *xybins[i:i + 2])
#         for j in range(nbins_xy):
#             index +=1
#             #print(f' bin =({i},{j});  index = {index}')
#             if k%25 ==0:
#                 k=0
#                 fig = plt.figure(figsize=figsize)
#             ax = fig.add_subplot(5, 5, k+1)
#             k+=1
#             sel_y = in_range(kre.Y, *xybins[j:j + 2])
#             sel   = sel_x & sel_y
#             entries = np.count_nonzero(sel)
#             if entries < min_entries:
#                 print(f'entries ={entries} not enough  to fit bin (i,j) =({i},{j})')
#                 valid [i, j] = False
#                 continue
#
#             try:
#                 z = kre.Z[sel]
#                 t = kre.E[sel]
#                 if fit_var == "Q":
#                     t = kre.Q[sel]
#
#                 x, y, yu = fitf.profileX(z, t, nbins_z, zrange)
#                 ax.errorbar(x, y, yu, np.diff(x)[0]/2,
#                              fmt="kp", ms=7, lw=3)
#                 seed = expo_seed(x, y)
#                 f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
#                 plt.plot(x, f.fn(x), "r-", lw=4)
#                 plt.grid(True)
#                 re = np.abs(f.errors[1] / f.values[1])
#                 #print(f' E +- Eu = {f.values[0]} +- {f.errors[0]}')
#                 #print(f' LT +- LTu = {-f.values[1]} +- {f.errors[1]}')
#                 #print(f' LTu/LT = {re} chi2 = {f.chi2}')
#
#                 if re > 0.5:
#                     print(f'Relative error to large, re ={re} for bin (i,j) =({i},{j})')
#                     print(f' LT +- LTu = {-f.values[1]} +- {f.errors[1]}')
#                     print(f' LTu/LT = {re} chi2 = {f.chi2}')
#                     valid [i, j] = False
#
#                 const [i, j] = f.values[0]
#                 constu[i, j] = f.errors[0]
#                 slope [i, j] = -f.values[1]
#                 slopeu[i, j] = f.errors[1]
#                 chi2  [i, j] = f.chi2
#                 valid [i, j] = True
#             except:
#                 print(f'fit failed for bin (i,j) =({i},{j})')
#                 pass
#     plt.tight_layout()
#
#     return KrLTSlices(Ez0  = Measurement(const, constu),
#                        LT   = Measurement(slope, slopeu),
#                        chi2 = chi2,
#                        valid = valid)
#
#
# def print_fit(krf: KrFit):
#     print(f' E (z=0) = {krf.par[0]} +-{krf.err[0]} ')
#     print(f' LT      = {krf.par[1]} +-{krf.err[1]} ')
#     print(f' chi2    = {krf.chi2} ')
#
#
# def print_krfit(f):
#     for i, val in enumerate(f.values):
#         print('fit par[{}] = {} error = {}'.format(i, val, f.errors[i]))
#
#
# def chi2(F, X, Y, SY):
#     fitx = F.fn(X)
#     n = len(F.values)
#     #print('degrees of freedom = {}'.format(n))
#     chi2t = 0
#     for i, x in enumerate(X):
#         chi2 = abs(Y[i] - fitx[i])/SY[i]
#         chi2t += chi2
#         #print('x = {} f(x) = {} y = {} ey = {} chi2 = {}'.format(
#                    #x, fitx[i], Y[i], SY[i], chi2 ))
#     return chi2t/(len(X)-n)
