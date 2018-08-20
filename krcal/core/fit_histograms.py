import numpy as np
import random

import matplotlib.pyplot as plt
from . histo_functions import h1
from . histo_functions import plot_histo
from . kr_types        import PlotLabels, FitFBPars
from   invisible_cities.evm.ic_containers  import Measurement


def gaussian_histo_example(mean, nevt, figsize=(10,10)):
    """Examples of histogramming gaussian distributions"""

    Nevt   = int(nevt)
    sigmas = np.random.uniform(low=1.0, high=10., size=4)

    fig       = plt.figure(figsize=figsize)
    pltLabels = PlotLabels(x='Energy', y='Events', title='Gaussian')

    e   = np.random.normal(100, sigmas[0], Nevt)
    ax  = fig.add_subplot(2, 2, 1)
    (_) = h1(e, bins=100, range=(mean - 5 * sigmas[0],mean + 5 * sigmas[0]))
    plot_histo(pltLabels, ax)

    pltLabels.title = 'Gaussian log scale'
    e   = np.random.normal(100, sigmas[1], Nevt)
    ax  = fig.add_subplot(2, 2, 2)
    (_) = h1(e, bins=100, range=(mean - 5 * sigmas[1],mean + 5 * sigmas[1]), log=True)
    plot_histo(pltLabels, ax, legendloc='upper left')

    pltLabels.title = 'Gaussian normalized'
    e    = np.random.normal(100, sigmas[2], Nevt)
    ax   = fig.add_subplot(2, 2, 3)
    (_)  = h1(e, bins=100, range=(mean - 5 * sigmas[2],mean + 5 * sigmas[2]), normed=True)
    plot_histo(pltLabels, ax, legendloc='upper right')

    pltLabels.title = 'Gaussian change histo pars'
    e    = np.random.normal(100, sigmas[3], Nevt)
    ax   = fig.add_subplot(2, 2, 4)
    (_)  = h1(e, bins=100, range=(mean - 5 * sigmas[3],mean + 5 * sigmas[3]),
                 color='red', width=2.0, style='dashed')
    plot_histo(pltLabels, ax, legendsize=14)

    plt.tight_layout()


def histo_gaussian_experiment_sample(exps, mexperiments, samples=9, canvas=(3,3),
                                     bins = 50, range_e = (9e+3,11e+3),
                                     figsize=(10,10)):
    """Takes an array of gaussian experiments and samples it"""
    fig = plt.figure(figsize=figsize)
    ks  = random.sample(range(mexperiments), samples)
    for i,k in enumerate(ks):
        ax  = fig.add_subplot(canvas[0], canvas[1], i+1)
        (_) = h1(exps[k], bins=bins, range = range_e)

    plt.tight_layout()


def histo_gaussian_params_and_pulls(mean, sigma, mus, umus, stds, ustds,
                                    bin_mus    = 50,
                                    bin_stds   = 50,
                                    bin_pull   = 50,
                                    range_mus  = (9950,10050),
                                    range_stds = (150,250),
                                    range_pull = (-5,5),
                                    figsize    =(10,10)):
    """Histograms mus and stds of gaussian experiments (values and pulls)"""
    fig       = plt.figure(figsize=figsize)

    ax        = fig.add_subplot(2, 2, 1)
    pltLabels = PlotLabels(x='mus', y='Events', title='mean')
    (_)       = h1(mus, bins=bin_mus, range=range_mus)
    plot_histo(pltLabels, ax)

    ax        = fig.add_subplot(2, 2, 2)
    (_)       = h1((mus-mean) / umus, bins=bin_pull, range=range_pull)
    pltLabels = PlotLabels(x='Pull(mean)', y='Events', title='Pull (mean)')
    plot_histo(pltLabels, ax)

    ax        = fig.add_subplot(2, 2, 3)
    pltLabels = PlotLabels(x='std ', y='Events', title='std')
    (_)       = h1(stds, bins=bin_stds, range=range_stds)
    plot_histo(pltLabels, ax)

    ax        = fig.add_subplot(2, 2, 4)
    (_)       = h1((stds-sigma) / ustds, bins=50, range=range_pull)
    pltLabels =PlotLabels(x='pull (std) ', y='Events', title='Pull (std)')
    plot_histo(pltLabels, ax)

    plt.tight_layout()


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


def histo_fit_fb_pars(fp, fpf = None, fpb = None,
                   range_chi2=(0,3),
                   range_e0 =(10000,12500),
                   range_lt=(2000, 3000)):

    ts, e0, lt, c2 = fp

    if fpf:
        ts, e0f, ltf, c2f = fpf
    if fpb:
        ts, e0b, ltb, c2b = fpb


    fig = plt.figure(figsize=(14,6))
    ax      = fig.add_subplot(1, 3, 1)
    _, _, c2_mu, c2_std        = h1(c2, bins=20, range = range_chi2, color='black', stats=True)
    plot_histo(PlotLabels('chi2','Entries',''), ax)
    if fpf:
        _, _, c2f_mu, c2f_std  = h1(c2f, bins=20, range = range_chi2, color='red', stats=True)
        plot_histo(PlotLabels('chi2','Entries',''), ax)
    if fpb:
        _, _, c2b_mu, c2b_std  = h1(c2b, bins=20, range = range_chi2, color='blue', stats=True)
        plot_histo(PlotLabels('chi2','Entries',''), ax)

    ax      = fig.add_subplot(1, 3, 2)
    _, _, e0_mu, e0_std       = h1(e0, bins=20, range = range_e0, color='black', stats=True)
    plot_histo(PlotLabels('E0','Entries',''), ax)
    if fpf:
        _, _, e0f_mu, e0f_std  = h1(e0f, bins=20, range = range_e0, color='red', stats=True)
        plot_histo(PlotLabels('E0','Entries',''), ax)
    if fpb:
        _, _, e0b_mu, e0b_std  = h1(e0b, bins=20, range = range_e0, color='blue', stats=True)
        plot_histo(PlotLabels('E0','Entries',''), ax)

    ax      = fig.add_subplot(1, 3, 3)
    _, _, lt_mu, lt_std        = h1(lt, bins=20, range = range_lt, color='black', stats=True)
    plot_histo(PlotLabels('LT','Entries',''), ax)
    if fpf:
        _, _, ltf_mu, ltf_std  = h1(ltf, bins=20, range = range_lt, color='red', stats=True)
        plot_histo(PlotLabels('LT','Entries',''), ax)
    if fpb:
        _, _, ltb_mu, ltb_std  = h1(ltb, bins=20, range = range_lt, color='blue', stats=True)
        plot_histo(PlotLabels('LT','Entries',''), ax)

    plt.tight_layout()

    return FitFBPars(Measurement(c2_mu, c2_std),
                     Measurement(c2f_mu, c2f_std),
                     Measurement(c2b_mu, c2b_std),
                     Measurement(e0_mu, e0_std),
                     Measurement(e0f_mu, e0f_std),
                     Measurement(e0b_mu, e0b_std),
                     Measurement(lt_mu, lt_std),
                     Measurement(ltf_mu, ltf_std),
                     Measurement(ltb_mu, ltb_std))



def histo_fit_sectors(fps,
                      range_chi2=(0,3),
                      range_e0 =(10000,12500),
                      range_lt=(2000, 3000)):

    fig = plt.figure(figsize=(14,6))

    C2 = []
    E0 = []
    LT  =[]

    ax      = fig.add_subplot(1, 3, 1)
    for fp in fps:
        ts, e0, lt, c2 = fp
        _, _, c2_mu, c2_std        = h1(c2, bins=20, range = range_chi2, color=None, stats=False)
        plot_histo(PlotLabels('Chi2','Entries',''), ax)
        C2.append(Measurement(c2_mu, c2_std))

    ax      = fig.add_subplot(1, 3, 2)
    for fp in fps:
        ts, e0, lt, c2 = fp
        _, _, e0_mu, e0_std       = h1(e0, bins=20, range = range_e0, color=None, stats=False)
        plot_histo(PlotLabels('E0','Entries',''), ax)
        E0.append(Measurement(e0_mu, e0_std))


    ax      = fig.add_subplot(1, 3, 3)
    for fp in fps:
        ts, e0, lt, c2 = fp
        _, _, lt_mu, lt_std        = h1(lt, bins=20, range = range_lt, color=None, stats=False)
        plot_histo(PlotLabels('LT','Entries',''), ax)
        LT.append(Measurement(lt_mu, lt_std))

    plt.tight_layout()

    return C2, E0, LT


def print_fit_sectors_pars(C2, E0, LT) :
    i=0
    for c2, e0, lt in zip(C2, E0, LT):
        xc2  = f'{c2.value:8.2f} +- {c2.uncertainty:6.2f};'
        xe0  = f'{e0.value:8.2f} +- {e0.uncertainty:6.2f};'
        xlt  = f'{lt.value:8.2f} +- {lt.uncertainty:6.2f};'

        print(f'wedge = {i}: chi2 = {xc2} e0 = {xe0} lt = {xlt}')
        i+=1


def print_fit_fb_pars(fbp : FitFBPars) :
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

def plot_fit_fb_pars(fp, fpf=None, fpb=None, ltlim=(2000,3000), e0lim=(11000,12000)):

    ts, e0, lt, c2 = fp

    if fpf:
        ts, e0f, ltf, c2f = fpf
    if fpb:
        ts, e0b, ltb, c2b = fpb

    fig = plt.figure(figsize=(14,6))
    ax      = fig.add_subplot(1, 2, 1)
    plt.errorbar(ts, lt, np.sqrt(lt), fmt="p", c="k")
    if fpb:
        plt.errorbar(ts, ltb, np.sqrt(ltb), fmt="p", c="b")
    if fpf:
        plt.errorbar(ts, ltf, np.sqrt(ltf), fmt="p", c="r")
    plt.ylim(*ltlim)
    plt.title('lifetime (t) ')


    ax      = fig.add_subplot(1, 2, 2)
    plt.errorbar(ts, e0, np.sqrt(e0), fmt="p", c="k")
    if fpb:
        plt.errorbar(ts, e0b, np.sqrt(e0b), fmt="p", c="b")
    if fpf:
        plt.errorbar(ts, e0f, np.sqrt(e0f), fmt="p", c="r")

    plt.ylim(*e0lim)
    plt.title('E0z (t)')
    plt.tight_layout()


def plot_fit_sectors(fps,  ltlim=(2000,3000), e0lim=(11000,12000)):

    fig = plt.figure(figsize=(14,6))

    C2 = []
    E0 = []
    LT  =[]

    ax      = fig.add_subplot(1, 2, 1)
    for fp in fps:
        ts, e0, lt, c2 = fp
        plt.errorbar(ts, lt, np.sqrt(lt), fmt="p")
    plt.ylim(*ltlim)
    plt.title('lifetime (t) ')

    ax      = fig.add_subplot(1, 2, 2)
    for fp in fps:
        ts, e0, lt, c2 = fp
        plt.errorbar(ts, e0, np.sqrt(e0), fmt="p")
    plt.ylim(*e0lim)
    plt.title('E0z (t)')

    plt.tight_layout()
