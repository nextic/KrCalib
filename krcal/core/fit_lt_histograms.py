import numpy as np
import random

from typing import Optional

import matplotlib.pyplot as plt
from . histo_functions import h1
from . histo_functions import plot_histo
from . core_functions  import value_from_measurement
from . kr_types        import PlotLabels
from . kr_types        import FitParTS, FitParFB
from   invisible_cities.evm.ic_containers  import Measurement
from typing  import Dict, List, Tuple, Sequence, Iterable


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


def histo_fit_fb_pars(fp         : FitParTS,
                      fpf        : Optional[FitParTS] = None,
                      fpb        : Optional[FitParTS] = None,
                      range_chi2 : Tuple[float, float] =(0,3),
                      range_e0   : Tuple[float, float] =(10000,12500),
                      range_lt   : Tuple[float, float] =(2000, 3000)) ->FitParFB:

    fig = plt.figure(figsize=(14,6))

    ax      = fig.add_subplot(1, 3, 1)
    _, _, c2_mu, c2_std        = h1(fp.c2, bins=20, range = range_chi2, color='black', stats=True)
    plot_histo(PlotLabels('chi2','Entries',''), ax)
    if fpf:
        _, _, c2f_mu, c2f_std  = h1(fpf.c2, bins=20, range = range_chi2, color='red', stats=True)
        plot_histo(PlotLabels('chi2','Entries',''), ax)
    if fpb:
        _, _, c2b_mu, c2b_std  = h1(fpb.c2, bins=20, range = range_chi2, color='blue', stats=True)
        plot_histo(PlotLabels('chi2','Entries',''), ax)

    ax      = fig.add_subplot(1, 3, 2)
    _, _, e0_mu, e0_std       = h1(fp.e0, bins=20, range = range_e0, color='black', stats=True)
    plot_histo(PlotLabels('E0','Entries',''), ax)
    if fpf:
        _, _, e0f_mu, e0f_std  = h1(fpf.e0, bins=20, range = range_e0, color='red', stats=True)
        plot_histo(PlotLabels('E0','Entries',''), ax)
    if fpb:
        _, _, e0b_mu, e0b_std  = h1(fpb.e0, bins=20, range = range_e0, color='blue', stats=True)
        plot_histo(PlotLabels('E0','Entries',''), ax)

    ax      = fig.add_subplot(1, 3, 3)
    _, _, lt_mu, lt_std        = h1(fp.lt, bins=20, range = range_lt, color='black', stats=True)
    plot_histo(PlotLabels('LT','Entries',''), ax)
    if fpf:
        _, _, ltf_mu, ltf_std  = h1(fpf.lt, bins=20, range = range_lt, color='red', stats=True)
        plot_histo(PlotLabels('LT','Entries',''), ax)
    if fpb:
        _, _, ltb_mu, ltb_std  = h1(fpb.lt, bins=20, range = range_lt, color='blue', stats=True)
        plot_histo(PlotLabels('LT','Entries',''), ax)

    plt.tight_layout()

    return FitParFB(Measurement(c2_mu, c2_std),
                     Measurement(c2f_mu, c2f_std),
                     Measurement(c2b_mu, c2b_std),
                     Measurement(e0_mu, e0_std),
                     Measurement(e0f_mu, e0f_std),
                     Measurement(e0b_mu, e0b_std),
                     Measurement(lt_mu, lt_std),
                     Measurement(ltf_mu, ltf_std),
                     Measurement(ltb_mu, ltb_std))


def plot_fit_fb_pars(fp    : FitParTS,
                     fpf   : Optional[FitParTS]  = None,
                     fpb   : Optional[FitParTS]  = None,
                     ltlim : Tuple[float, float] = (1500,3000),
                     e0lim : Tuple[float, float] = (9000,12500)):

    fig = plt.figure(figsize=(14,6))
    ax      = fig.add_subplot(1, 2, 1)
    plt.errorbar(fp.ts, fp.lt, np.sqrt(fp.lt), fmt="p", c="k")
    if fpb:
        plt.errorbar(fpb.ts, fpb.lt, np.sqrt(fpb.lt), fmt="p", c="b")
    if fpf:
        plt.errorbar(fpf.ts, fpf.lt, np.sqrt(fpf.lt), fmt="p", c="r")
    plt.ylim(*ltlim)
    plt.title('lifetime (t) ')

    ax      = fig.add_subplot(1, 2, 2)
    plt.errorbar(fp.ts, fp.e0, np.sqrt(fp.e0), fmt="p", c="k")
    if fpb:
        plt.errorbar(fpb.ts, fpb.e0, np.sqrt(fpb.e0), fmt="p", c="b")
    if fpf:
        plt.errorbar(fpf.ts, fpf.e0, np.sqrt(fpf.e0), fmt="p", c="r")

    plt.ylim(*e0lim)
    plt.title('E0z (t)')
    plt.tight_layout()


def histo_fit_sectors(fps : Iterable[FitParTS],
                      range_chi2 : Tuple[float, float] =(0,3),
                      range_e0   : Tuple[float, float] =(10000,12500),
                      range_lt   : Tuple[float, float] =(2000, 3000)) ->FitParTS:

    fig = plt.figure(figsize=(14,6))

    C2  = []
    E0  = []
    LT  = []
    C2u = []
    E0u = []
    LTu = []

    ax      = fig.add_subplot(1, 3, 1)
    for fp in fps:
        _, _, c2_mu, c2_std        = h1(fp.c2, bins=20, range = range_chi2, color=None, stats=False)
        plot_histo(PlotLabels('Chi2','Entries',''), ax)
        C2.append(c2_mu)
        C2u.append(c2_std)

    ax      = fig.add_subplot(1, 3, 2)
    for fp in fps:
        _, _, e0_mu, e0_std       = h1(fp.e0, bins=20, range = range_e0, color=None, stats=False)
        plot_histo(PlotLabels('E0','Entries',''), ax)
        E0.append(e0_mu)
        E0u.append(e0_std)

    ax      = fig.add_subplot(1, 3, 3)
    for fp in fps:
        _, _, lt_mu, lt_std        = h1(fp.lt, bins=20, range = range_lt, color=None, stats=False)
        plot_histo(PlotLabels('LT','Entries',''), ax)
        LT.append(lt_mu)
        LTu.append(lt_std)

    plt.tight_layout()

    return FitParTS(ts  = np.array(range(len(fps))),
                    e0  = np.array(E0),
                    lt  = np.array(LT),
                    c2  = np.array(C2),
                    e0u = np.array(E0u),
                    ltu = np.array(LTu))


def plot_fit_sectors(fps : Iterable[FitParTS],
                     range_chi2 : Tuple[float, float] =(0,3),
                     range_e0   : Tuple[float, float] =(10000,12500),
                     range_lt   : Tuple[float, float] =(2000, 3000)):

    fig = plt.figure(figsize=(14,6))

    ax      = fig.add_subplot(1, 2, 1)
    for fp in fps:
        plt.errorbar(fp.ts, fp.lt, np.sqrt(fp.lt), fmt="p")
    plt.ylim(*range_lt)
    plt.title('lifetime (t) ')

    ax      = fig.add_subplot(1, 2, 2)
    for fp in fps:
        plt.errorbar(fp.ts, fp.e0, np.sqrt(fp.e0), fmt="p")
    plt.ylim(*range_e0)
    plt.title('E0z (t)')

    plt.tight_layout()


def print_fit_sectors_pars(fpts : FitParTS) :

    for i, c2, e0, lt, e0u, ltu in zip(fpts.ts,
                                            fpts.c2,
                                            fpts.e0,
                                            fpts.lt,
                                            fpts.e0u,
                                            fpts.ltu):

        xc2  = f'{c2:8.2f} ;'
        xe0  = f'{e0:8.2f} +- {e0u:6.2f};'
        xlt  = f'{lt:8.2f} +- {ltu:6.2f};'

        print(f'wedge = {i}: chi2 = {xc2} e0 = {xe0} lt = {xlt}')


def print_fit_fb_pars(fbp : FitParFB) :
    xc2  = f'full  = {fbp.c2.value:8.2f}  +- {fbp.c2.uncertainty:6.2f};'
    xc2f = f'front = {fbp.c2f.value:8.2f} +- {fbp.c2f.uncertainty:6.2f};'
    xc2b = f'back  = {fbp.c2b.value:8.2f} +- {fbp.c2b.uncertainty:6.2f};'
    xe0  = f'full  = {fbp.e0.value:8.2f}  +- {fbp.e0.uncertainty:6.2f};'
    xe0f = f'front = {fbp.e0f.value:8.2f} +- {fbp.e0f.uncertainty:6.2f};'
    xe0b = f'back  = {fbp.e0b.value:8.2f} +- {fbp.e0b.uncertainty:6.2f};'
    xlt  = f'full  = {fbp.lt.value:8.2f}  +- {fbp.lt.uncertainty:6.2f};'
    xltf = f'front = {fbp.ltf.value:8.2f} +- {fbp.ltf.uncertainty:6.2f};'
    xltb = f'back  = {fbp.ltb.value:8.2f} +- {fbp.ltb.uncertainty:6.2f};'

    print(f'chi2:  {xc2} {xc2f} {xc2b}')
    print(f'e0  :  {xe0} {xe0f} {xe0b}')
    print(f'lt  :  {xlt} {xltf} {xltb}')
