#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 2018

Plots functions for Kr,

@author: G. Martinez, J.A.hernando
"""

import numpy             as np
#import scipy.stats       as stats
import matplotlib.pyplot as plt
import scipy.optimize as optimize

import tables            as tb
import krcal.utils.hst_extend_functions   as hst

from collections import namedtuple
from   invisible_cities.evm  .ic_containers  import Measurement
#from dataclasses import dataclass

import invisible_cities.core.fit_functions as fitf

from invisible_cities.core .core_functions import in_range
from invisible_cities.icaro.hst_functions  import shift_to_bin_centers

from krcal.core.fit_functions import fit_slices_2d_expo
from krcal.core.fit_functions import fit_slices_1d_gauss
from krcal.core.fit_functions import expo_seed
from krcal.core.fit_functions import to_relative
from invisible_cities.icaro.hst_functions import labels


default_cmap = 'jet'


XYMap = namedtuple('XYMap',
                   ('x', 'y', 'value', 'uncertainty', 'valid'))


def selection_in_band(E, Z, Erange, Zrange, Zfitrange, nsigma = 3.5,
                      Znbins = 50, Enbins =100, plot=True):
    """ This returns a selection of the events that are inside the Kr E vz Z
    returns: np.array(bool)
        If plot=True, it draws E vs Z and the band
    """
    Zfit = Zfitrange

    Zbins = np.linspace(*Zrange, Znbins + 1)
    Ebins = np.linspace(*Erange, Enbins + 1)

    Zcenters = shift_to_bin_centers(Zbins)
    Zerror   = np.diff(Zbins) * 0.5

    sel_e = in_range(E, *Erange)
    mean, sigma, chi2, ok = fit_slices_1d_gauss(Z[sel_e], E[sel_e], Zbins, Ebins, min_entries=5e2)
    ok = ok & in_range(Zcenters, *Zfit)

    def _line_cut(sign):
        x         = Zcenters[ok]
        y         = mean.value[ok] + sign*nsigma * sigma.value[ok]
        yu        = mean.uncertainty[ok]
        seed      = expo_seed(x, y)
        efit  = fitf.fit(fitf.expo, x, y, seed, sigma=yu)
        assert np.all(efit.values != seed)
        return efit.fn

    lowE_cut  = _line_cut(-1.)
    highE_cut = _line_cut(+1.)

    sel_inband = in_range(E, lowE_cut(Z), highE_cut(Z))

    if (plot == False): return sel_inband


    plt.hist2d  (Z, E, (Zbins, Ebins), cmap=default_cmap)
    plt.errorbar(   Zcenters[ok], mean.value[ok],
                sigma.value[ok],     Zerror[ok],
                "kp", label="Kr peak energy $\pm 1 \sigma$")
    f = fitf.fit(fitf.expo, Zcenters[ok], mean.value[ok], (1e4, -1e3))
    plt.plot(Zcenters, f.fn(Zcenters), "r-")
    print(f.values)
    plt.plot    (Zbins,  lowE_cut(Zbins),  "m", lw=2, label="$\pm "+str(nsigma)+" \sigma$ region")
    plt.plot    (Zbins, highE_cut(Zbins),  "m", lw=2)
    plt.legend()
    labels("Drift time (µs)", "S2 energy (pes)", "Energy vs drift")

    return sel_inband


#--- functions relatwd tiwh binned fit


def lt(z, v, znbins, zrange, vnbins, vrange, plot = True):
    """ compute the lifetime of v-variable (S2e, Se1, E) vs Z
    """
    zbins     = np.linspace(* zrange,  znbins + 1)
    vbins     = np.linspace(* vrange,  vnbins + 1)
    x, y, yu = fitf.profileX(z, v, znbins, zrange)
    seed = expo_seed(x, y)
    f    = fitf.fit(fitf.expo, x, y, seed, sigma=yu)

    if (not plot): return f

    #print('energy_0', f.values[0], ' +- ', f.errors[0] )
    #print('lifetime', f.values[1], ' +- ', f.errors[1] )

    frame_data = plt.gcf().add_axes((.1, .3,
                                 .8, .6))
    plt.hist2d(z, v, (zbins, vbins))
    x, y, yu = fitf.profileX(z, v, znbins, zrange)
    plt.errorbar(x, y, yu, np.diff(x)[0]/2, fmt="kp", ms=7, lw=3)
    plt.plot(x, f.fn(x), "r-", lw=4)
    frame_data.set_xticklabels([])
    labels("", "Energy (pes)", "Lifetime fit")
    lims = plt.xlim()
    frame_res = plt.gcf().add_axes((.1, .1,
                                    .8, .2))
    plt.errorbar(x, (f.fn(x) - y) / yu, 1, np.diff(x)[0] / 2, fmt="p", c="k")
    plt.plot(lims, (0, 0), "g--")
    plt.xlim(*lims)
    plt.ylim(-5, +5)
    labels("Drift time (µs)", "Standarized residual")
    return f

def lt_vs_t(z, v, t, znbins, zrange, vnbins, vrange, tnbins, trange):
    """ returns the profile-fit to the lifetime in slices of time
    """
    tbins  = np.linspace(* trange,  tnbins + 1)
    fs = []
    for i in range(tnbins):
        sel = in_range(t, tbins[i], tbins[i+1])
        fi = lt(z[sel], v[sel], znbins, zrange, vnbins, vrange, plot = False)
        fs.append(fi)
    return fs


def ltmap(X, Y, Z, E, XYbins, Znbins, Zrange):
    """ returns E0 and LT from the lifetime fit
    """
    Escale, ELT,\
    Echi2, Eok = fit_slices_2d_expo(X, Y, Z, E, XYbins, XYbins, Znbins, zrange=Zrange, min_entries=50)
    Eok        = Eok & (ELT.value < -100) & (ELT.value > -1e5) & np.isfinite(Echi2)
    Escale_rel = to_relative(Escale, percentual=True)
    ELT_rel    = to_relative(   ELT, percentual=True)
    # xs, ys = 0.5*(XYbins[:-1]+XYbins[1:]), 0.5*(XYbins[:-1]+XYbins[1:])
    # e0map = XYMap(xs, ys,  Escale_rel.value, Escale_rel.uncertainty, Eok)
    # ltmap = XYMap(xs, ys, -ELT_rel.value   , ELT_rel.uncertainty   , Eok)
    # return e0map, ltmap
    ELT = Measurement(-1.*ELT.value, ELT.uncertainty)
    return Escale, ELT, Echi2, Eok


def ltmap_vs_t(X, Y, Z, E, T, XYbins, Znbins, Zrange, Tnbins, Trange):
    """ returns the LT XYmap and Geo XYMap in time T intervals
    """
    xs, ys = 0.5*(XYbins[:-1]+XYbins[1:]), 0.5*(XYbins[:-1]+XYbins[1:])
        # e0map = XYMap(xs, ys,  Escale_rel.value, Escale_rel.uncertainty, Eok)
        # ltmap = XYMap(xs, ys, -ELT_rel.value   , ELT_rel.uncertainty   , Eok)
        # return e0map, ltmap
    tbins  = np.linspace(* Trange,  Tnbins + 1)
    e0maps, ltmaps = [], []
    for i in range(Tnbins):
        sel = in_range(T, tbins[i], tbins[i+1])
        Escale, ELT, Echi2, Eok = ltmap(X[sel], Y[sel], Z[sel], E[sel], XYbins, Znbins, Zrange)
        e0map = XYMap(xs, ys, Escale.value, Escale.uncertainty, Eok); e0map.chi2 = Echi2
        ltmap = XYMap(xs, ys, ELT.value   , ELT.uncertainty   , Eok); ltmap.chi2 = Echi2
        e0maps.append(ie0map); ltmaps.append(iltmap)
    return e0maps, ltmaps


#---- Functions related with unbinned lifetime fit


def lt_lsqfit(Z, E, chi2 = True, nbins = 12):
    """ unbinned fit to the lifetime, return e0, lt best estimate, chi2, and valid (bool) flag
    if chi2 is False, return 0 for chi2,
    nbins is the number of bins to compute the chi2 (default 12)
    """
    ok = True
    e0, lt, e0u, ltu, xchi2 = 0, 0, 0, 0, 0
    DE = - np.log(E)
    try:
        cc, cov = np.polyfit(Z, DE, 1, full = False, cov = True )
        #print(cov)
        a, b = cc[0], cc[1]
        lt = 1/a       ; ltu = lt*lt *np.sqrt(cov[0, 0])
        e0 = np.exp(-b); e0u = e0    *np.sqrt(cov[1, 1])
    except:
        ok = False
        pass
    # print('lifetime :', lt, '+-', ult)
    # print('e0       :', e0, '+-', ue0)
    me0, mlt = Measurement(e0, e0u), Measurement(lt, ltu)

    if (not chi2 or not ok):
        return me0, mlt, xchi2, ok

    xs, ys, uys = fitf.profileX(Z, DE, nbins)
    cok = ~np.isnan(uys)
    try:
        res = (a*xs[cok]+b-ys[cok])/uys[cok]
        # print(res, uys)
        xchi2 = np.sum(res*res)/(1.*len(xs)-1)
    except:
        ok = False
        pass
    #print(me0, mlt, chi2)
    return me0, mlt, xchi2, ok


def lt_vs_t_lsqfit(Z, E, T, Tbins, nbins = 12):
    """ returns the unbinned fit to the lifetime in slices of time
    """
    fs = []
    for i in range(len(Tbins)-1):
        sel = in_range(T, Tbins[i], Tbins[i+1])
        fi = lt_lsqfit(Z[sel], E[sel], nbins = nbins)
        fs.append(fi)
    return fs


def ltmap_lsqfit(X, Y, Z, E, XYbins, min_entries = 20, nbins = 10):
    """ compute the lifetime map with the unbinned fit. Returns E0, LT measureents matrices (x, y)
    chi2 matrix and valid boolean matrix.
    """
    nbins   = len(XYbins)-1
    e0      = np.zeros(nbins*nbins).reshape(nbins, nbins)
    lt      = np.zeros(nbins*nbins).reshape(nbins, nbins)
    e0u     = np.zeros(nbins*nbins).reshape(nbins, nbins)
    ltu     = np.zeros(nbins*nbins).reshape(nbins, nbins)
    chi2    = np.zeros(nbins*nbins).reshape(nbins, nbins)
    valid   = np.zeros(nbins*nbins, dtype=bool).reshape(nbins, nbins)

    for i in range(nbins):
        sel_x = in_range(X, *XYbins[i: i+2])
        for j in range(nbins):
            sel_y = in_range(Y, *XYbins[j: j+2])
            sel   = sel_x & sel_y
            if np.count_nonzero(sel) < min_entries: continue

            try:
                me0, mlt, ichi2, iok = lt_lsqfit(Z[sel], E[sel], chi2=True, nbins=nbins)
                e0 [i, j] = me0.value
                e0u[i, j] = me0.uncertainty
                lt [i, j] = mlt.value
                ltu[i, j] = mlt.uncertainty
                chi2  [i, j] = ichi2
                valid [i, j] = iok
            except:
                pass
    return Measurement(e0, e0u), Measurement(lt, ltu), chi2, valid


def ltmap_vs_t_lsqfit(X, Y, Z, E, T, XYbins, Tbins, min_entries = 20):
    """ returns the LT-maps and Geo-maps in time T intervals using the unbinned fit
    """
    fs = []
    for i in range(len(Tbins)-1):
        sel = in_range(T, Tbins[i], Tbins[i+1])
        fi = ltmap_lsqfit(X[sel], Y[sel], Z[sel], E[sel], XYbins, min_entries = min_entries)
        fs.append(fi)
    return fs

#--- fit the lifetime in regions of a variable an time

def lt_vs_t_vs_v_lsqfit(Z, E, T, V, Tbins, Vbins, nbins = 12):
    """ it retuns the result of the lifetime fit in bins of time and in bins of a V variable (i.e Radius)
    """
    fs = []
    for i in range(len(Vbins)-1):
        sel = in_range(V, Vbins[i], Vbins[i+1])
        fi = lt_vs_t_lsqfit(Z[sel], E[sel], T[sel], Tbins, nbins = nbins)
        fs.append(fi)
    return fs

#---- Functions related with e0maps

def xymap_vprofile_zslice(V, X, Y, Z, Zbins, XYnbins, XYrange, std = True):
    vs, vus, voks = [], [], []
    for ii in range(len(Zbins)-1):
        sel_ii = in_range(Z, Zbins[ii], Zbins[ii+1])
        x, y, z, uz = fitf.profileXY(X[sel_ii], Y[sel_ii], V[sel_ii], XYnbins, XYnbins,
                                                 xrange=XYrange, yrange=XYrange, std = std)
        ok = z >0
        vs.append(z); vus.append(uz); voks.append(ok)
    return vs, vus, voks

def xymap_mean_std(xymap, ok):
    return np.mean(xymap[ok].flatten()), np.std(xymap[ok].flatten())

#---- Functions to compare two Measurement arrays (two maps)

def xymap_compare(xymap1, xymap0, mask1, mask0, type = 'difference', default = 0.):
    """ compare two (x, y) maps, xymap1, xmap0 are the arrays (x, y), mask1, mask1 are the array mask
    Returns ans array with the differencei (or pull, 100*ratio) and a mask bool array.
    """
    # types: pull, ratio

    v0, uv0, ok0 = xymap0.value, xymap0.uncertainty, mask0
    v1, uv1, ok1 = xymap1.value, xymap1.uncertainty, mask1

    ok  = np.logical_and(ok0, ok1)
    nbins, nbins = v0.shape

    d   = default * np.ones (nbins * nbins).reshape(nbins, nbins)
    err =           np.zeros(nbins * nbins).reshape(nbins, nbins)

    err = np.sqrt(uv0*uv0 + uv1*uv1)
    ok = np.logical_and(ok, err > 0.)

    d[ok] = v1[ok]-v0[ok]
    err[~ok] = default

    if (type == 'ratio'):
        sel = v0 > 0
        d[sel]   = 100.*d[sel]/v0[sel]
        err[sel] = 100.*err[sel]/v0[sel]
    if (type == 'pull'):
        sel = err > 0.
        d[sel] = d[sel] / err[sel]

    dmap = Measurement(d, err)
    return dmap, ok


def xymap_add(xymap0, xymap1):

    x, y         = xymap0.x    , xymap0.y
    v0, uv0, ok0 = xymap0.value, xymap0.uncertainty, xymap0.valid
    v1, uv1, ok1 = xymap1.value, xymap1.uncertainty, xymap1.valid

    ok  = np.logical_and(ok0, ok1)
    nbins, nbins = v0.shape

    err = np.zeros(nbins*nbins).reshape(nbins, nbins)
    err[ok] = 1./np.sqrt(1./(uv0[ok]*uv0[ok])+1./(uv1[ok]*uv1[ok]))
    err_m = np.mean(err[ok].flatten())

    val = np.zeros(nbins*nbins).reshape(nbins, nbins)
    val[ok] = (v0[ok]/(uv0[ok]*uv0[ok]) + v1[ok]/(uv1[ok]*uv1[ok]))*err[ok]*err[ok]
    val_m = np.mean(val[ok].flatten())

    val[~ok] = val_m
    err[~ok] = -1.*err_m

    amap = XYMap(x, y, val, err, ok)
    return amap


#---- Functions related with gauss fit and energy resolution

import scipy.optimize as optimize
def fgauss(x, a, b, c):
    return a*np.exp(-(x-b)**2/(2*c**2))

def hgaussfit(V, Vnbins, Vrange):
    ys, xs = np.histogram(V, Vnbins, Vrange)
    xs = 0.5*(xs[1:]+xs[:-1])
    p0s = (np.sum(ys), np.mean(V), np.std(V))
    fpar, fcov = optimize.curve_fit(fgauss, xs, ys, p0s)
    return fpar, np.sqrt(np.diag(fcov))

def fwhm(fpar, ferr):
    r  = 2.355*100.*fpar[2]/fpar[1]
    ru = r*np.sqrt(ferr[1]/fpar[1]**2 + ferr[2]/fpar[2]**2)
    return Measurement(r, ru)

def resolution(V, Vnbins, Vrange):
    fpar, ferr = hgaussfit(V, Vnbins, Vrange)
    eres = fwhm(fpar, ferr)
    label = 'resolution {0:4.2f} $\pm$ {1:4.2f} %'.format(eres.value, eres.uncertainty)
    return eres, label
