"""Module fit_energy_functions.
This module includes the fit functions for the energy.

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
from typing      import Tuple, Iterable
import warnings

from   invisible_cities.core.core_functions import in_range


from pandas import DataFrame

from . fit_energy_functions import fit_energy

from ..core. kr_types import FitCollection

from . plt_functions import plot_histo
from ..core. stat_functions       import relative_error_ratio




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
                plot_histo('E','Entries',f' 0 < R < {r} 0 < z < {z}', ax, legend= True,
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
                        r_range: Tuple[float,float] = (3.5, 4.5),
                        figsize = (14,10)):

    def extrapolate_to_qbb(es : float)->float:
        return np.sqrt(41 / 2458) * es

    def np_extrapolate_to_qbb(es : np.array)->np.array:
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
    ax.set_ylim(r_range)
    ax2.set_ylim(np_extrapolate_to_qbb(np.array(r_range)))

    ax.set_xlabel(' z (mm)')
    ax.set_ylabel('resolution FWHM (%)')
    ax2.set_ylabel('resolution Qbb FWHM (%)')

    plt.legend()
    plt.show()

