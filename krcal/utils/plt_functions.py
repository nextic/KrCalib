#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 2018

Plots functions for Kr,

@author: G. Martinez, J.A.hernando
"""

import numpy                as np
import matplotlib.pyplot    as plt
import krcal.utils.hst_functions   as hst
import krcal.dev.akr_functions     as akr

import invisible_cities.core.fit_functions as fitf
from invisible_cities.icaro.hst_functions import labels
from invisible_cities.core .core_functions import in_range

cmap_default = 'jet'

#--- plotting 1D histogram of 2 dsts

def double_hist(h1, h2, binning, label0="Original", label1="Filtered", **kwargs):
    plt.hist(h1, binning, label=label0, alpha=0.5, color="g", **kwargs)
    plt.hist(h2, binning, label=label1, alpha=0.5, color="m", **kwargs)
    plt.legend()

def dst_compare_vars(dst1, dst2):
    dst    = dst1
    subdst = dst2

    plt.figure(figsize=(20, 15))

    plt.subplot(3, 4, 1)
    double_hist(dst.nS2, subdst.nS2, np.linspace(0, 5, 6))
    plt.yscale("log")
    labels("Number of S2s", "Entries", "# S2")

    plt.subplot(3, 4, 2)
    double_hist(dst.S1e, subdst.S1e, np.linspace(0, 50, 51))
    labels("S1 integral (pes)", "Entries", "S1 energy")

    plt.subplot(3, 4, 3)
    double_hist(dst.S1w, subdst.S1w, np.linspace(0, 600, 25))
    labels("S1 width (ns)", "Entries", "S1 width")

    plt.subplot(3, 4, 4)
    double_hist(dst.S1h, subdst.S1h, np.linspace(0, 15, 31))
    labels("S1 height (pes)", "Entries", "S1 height")

    plt.subplot(3, 4, 5)
    double_hist(dst.Nsipm, subdst.Nsipm, np.linspace(0, 100, 51))
    labels("Number of SiPMs", "Entries", "# SiPMs")

    plt.subplot(3, 4, 6)
    double_hist(dst.S2e, subdst.S2e, np.linspace(0, 25e3, 101))
    labels("S2 integral (pes)", "Entries", "S2 energy")

    plt.subplot(3, 4, 7)
    double_hist(dst.S2w, subdst.S2w, np.linspace(0, 50, 26))
    labels("S2 width (µs)", "Entries", "S2 width")

    plt.subplot(3, 4, 8)
    double_hist(dst.S2h, subdst.S2h, np.linspace(0, 1e4, 101))
    labels("S2 height (pes)", "Entries", "S2 height")

    plt.subplot(3, 4, 9)
    double_hist(dst.Z, subdst.Z, np.linspace(0, 600, 101))
    labels("Drift time (µs)", "Entries", "Drift time")

    plt.subplot(3, 4, 10)
    double_hist(dst.X, subdst.X, np.linspace(-200, 200, 101))
    labels("X (mm)", "Entries", "X")

    plt.subplot(3, 4, 11)
    double_hist(dst.Y, subdst.Y, np.linspace(-200, 200, 101))
    labels("Y (mm)", "Entries", "Y")

    plt.subplot(3, 4, 12)
    double_hist(dst.S2q, subdst.S2q, np.linspace(0, 5e3, 101))
    labels("Q (pes)", "Entries", "S2 charge")

    plt.tight_layout()


#--- Plotting maps (a 2D array)

def plt_xymap(x, y, xymap, mask, vbins, vrange, label = ''):
    """ plot the xymap, the histogram of the values and the xymap
    """
    ok = mask
    c = hst.Canvas(1, 2)
    hst.hist(xymap[ok].flatten(), vbins, vrange, canvas=c(1), xylabels=(label, ''))
    hst.display_matrix(x, y, xymap, canvas=c(2),
                       cmin = vrange[0], cmax = vrange[1], cmap=cmap_default,
                       xylabels = ('x (mm)', 'y (mm)', label))
    plt.clim(*vrange)
    plt.tight_layout()
    return c

def plt_var_xymap_wslice(V, X, Y, W, Vnbins, Vrange, XYnbins, XYrange, Wnbins, Wrange, label = ''):
    """ plot the XY map of a variable V in slices of second W variable. Vrange fix the range of the xymap
    """
    vmin, vmax = Vrange
    c = hst.Canvas(Vnbins/2, Vnbins/2)
    zzs = np.linspace(Wrange[0], Wrange[1], Wnbins+1)
    for ii in range(len(zzs)-1):
        sel = in_range(W, zzs[ii], zzs[ii+1])
        x, y, z, uz = fitf.profileXY(X[sel], Y[sel], V[sel], XYnbins, XYnbins,
                                     xrange=XYrange, yrange=XYrange)
        hst.display_matrix(x, y, z, cmap = default_cmap, canvas=c(ii+1), vmin = vmin, vmax=vmax,
                           xylabels=('x (mm)', 'y (mm)', label+ ' in ['+str(zzs[ii])+', '+str(zzs[ii+1])+']') );
        plt.tight_layout()
    return c

def plt_entries_xymap_wslice(X, Y, W, XYnbins, XYrange, Wnbins, Wrange, Crange, vname = ''):
    """ plot the number of entries in a XY map in slices of W variable,
    Crange fix the range of events of the xymap
    """
    wmin, wmax = Wrange
    c = hst.Canvas(Wnbins/2, Wnbins/2)
    zzs = np.linspace(wmin, wmax, Wnbins+1)
    for ii in range(len(zzs)-1):
        sel = in_range(W, zzs[ii], zzs[ii+1])
        hst.hist2d(X[sel], Y[sel], (XYnbins, XYnbins), (XYrange, XYrange),
                    cmap = cmap_default, canvas = c(ii+1),
                    xylabels = ('x (mm)', 'y (mm)',
                               'evts '+vname+' in ['+str(zzs[ii])+', '+str(zzs[ii+1])+']') );
        plt.colorbar().set_label("Number of events")
        plt.clim(*Crange)
        plt.tight_layout()
    return c

def plt_v_vs_u(V, U, Vnbins, Unbins, Vrange, Urange , Vname='', Uname=''):
    c = hst.Canvas(1, 2)
    hst.hist2d(U, V, (Unbins, Vnbins), (Urange, Vrange), canvas=c(1), xylabels=(Uname, Vname));
    xs, ys, eys = fitf.profileX(U, V, Unbins, xrange=Urange)
    hst.errorbar(xs, ys, yerr=eys, fmt='*', canvas=c(1), c='black')
    ymean = np.mean(ys)
    hst.hist(100.*ys/ymean, 20, canvas=c(2), xylabels=(' deviation ratio (%)', ''));
    return c

#--- Plotting with dates in the axis

def plt_xdates():
    plt.gca().xaxis.set_major_formatter(md.DateFormatter('%m-%d %H:%M'))
    #plt.gcf().autofmt_xdate()
    plt.xticks( rotation=25 )
    return

#--- string things

def str_range(Vname, Vrange, form = '3.0f'):
    s = '{0:'+form+'} < '+Vname+' < {1:'+form+'}'
    s = s.format(*Vrange)
    return s

#---- Plotting the Energy resolution

def plt_energy(V, Vbins, label = ''):
    Vnbins, Vrange = len(Vbins), (Vbins.min(), Vbins.max())
    Vcenters       = 0.5*(Vbins[:-1]+Vbins[1:])
    fpar, ferr = akr.hgaussfit(V, Vnbins, Vrange)
    eres = akr.fwhm(fpar, ferr)
    label = label + ': {0:4.2f} $\pm$ {1:4.2f} %'.format(eres.value, eres.uncertainty)
    plt.hist(V, Vnbins, Vrange, label=label)
    plt.plot(Vcenters, akr.fgauss(Vcenters, *fpar), color='black')
    plt.legend()


def plt_eresolution_zr(V, R, Z, Vnbins, Vrange, Rbins, Zbins):
    Zcenters = 0.5*(Zbins[0:-1]+Zbins[1:])
    for i in range(len(Rbins)-1):
        sel_i = in_range(R, Rbins[i], Rbins[i+1])
        es, eus = [], []
        label = '{0:2.0f} < R < {1:2.0f}'.format(Rbins[i], Rbins[i+1])
        for j in range(len(Zbins)-1):
            sel_ij = sel_i & in_range(Z, Zbins[j], Zbins[j+1])
            eres, _ = akr.resolution(V[sel_ij], Vnbins, Vrange)
            es .append(eres.value)
            eus.append(eres.uncertainty)
        plt.errorbar(Zcenters, es, eus, label = label, fmt='o', markersize=10., elinewidth=10.)
    plt.grid(True)
    plt.xlabel(' z (mm)')
    plt.ylabel('resolution (%)')
    plt.legend();

#def plt_u_var(u, v, range_u = None, range_v = None, n = 100, nxy = 20, uname='', vname=''):
#    c = hst.Canvas(1, 2)
#    if (not range_u):
#        range_u = (np.min(u)  , np.max(u))
#    if (not range_v):
#        range_v = (np.min(v), np.max(v))
#    hst.hist2d(u, v, (nxy, nxy), (range_u, range_v), canvas=c(1), xylabels=(uname, vname));
#    xs, ys, eys = fitf.profileX(u, v, n, xrange=range_u)
#    hst.errorbar(xs, ys, yerr=eys, fmt='*', canvas=c(1), c='black')
#    ymean = np.mean(ys)
#    hst.hist(100.*ys/ymean, 20, canvas=c(2), xylabels=(' deviation ratio (%)', ''));
#    return c
