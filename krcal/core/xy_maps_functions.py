"""Module xy_maps_functions.
This module includes the functions needed to draw xy maps.

Notes
-----
    KrCalib code depends on the IC library.
    Public functions are documented using numpy style convention

Documentation
-------------
    Insert documentation https
"""
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

import numpy as np
import pandas as pd
import seaborn as sns
from . kr_types        import MapType
from . kr_types        import ASectorMap

from typing            import List, Tuple, Dict, Sequence, Iterable
from typing            import Optional

from numpy import sqrt
from pandas import DataFrame

import logging
log = logging.getLogger()


def draw_xy_maps(aMap    : ASectorMap,
                 e0lims   : Optional[Tuple[float, float]] = None,
                 ltlims   : Optional[Tuple[float, float]] = None,
                 eulims   : Optional[Tuple[float, float]] = None,
                 lulims   : Optional[Tuple[float, float]] = None,
                 cmap    :  Optional[Colormap]            = None,
                 figsize : Tuple[float, float]            = (14,10)):
    """
    draws correction maps (e0, lt, e0u, ltu) in bins of xy.

    Parameters
    ----------
    aMap
        A container of maps.
        class ASectorMap:
            chi2  : DataFrame
            e0    : DataFrame
            lt    : DataFrame
            e0u   : DataFrame
            ltu   : DataFrame

    e0lims
        Defines the range of e0 in pes (e.g, (8000,14000)).
    ltlims
        Defines the range of lt in mus (e.g, (3000,5000)).
    eulims
        Defines the range of e0u in pes (or relative).
    lulims
        Defines the range of ltu in mus (or relative).
    cmap
        color map. For example: cmap = matplotlib.cm.viridis (defaults to seaborn)
    figsize
        Range definint the figure size.

    Returns
    -------
    Nothing
        Function produces a plot.

    """
    def vmin_max(lims):
        if lims == None:
            vmin = None
            vmax = None
        else:
            vmin=lims[0]
            vmax=lims[1]
        return vmin, vmax

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(2,2,1)
    vmin, vmax = vmin_max(e0lims)
    sns.heatmap(aMap.e0.fillna(0), vmin=vmin, vmax=vmax, cmap=cmap, square=True)

    ax = fig.add_subplot(2,2,2)
    vmin, vmax = vmin_max(eulims)
    sns.heatmap(aMap.e0u.fillna(0), vmin=vmin, vmax=vmax, cmap=cmap, square=True)

    ax = fig.add_subplot(2,2,3)
    vmin, vmax = vmin_max(ltlims)
    sns.heatmap(aMap.lt.fillna(0), vmin=vmin, vmax=vmax, cmap=cmap, square=True)

    ax = fig.add_subplot(2,2,4)
    vmin, vmax = vmin_max(lulims)
    sns.heatmap(aMap.ltu.fillna(0), vmin=vmin, vmax=vmax, cmap=cmap, square=True)
    plt.tight_layout()
    plt.show()


def draw_xy_map(aMap    : ASectorMap,
                wmap    : MapType,
                norm    : float                         = 1,
                alims   : Optional[Tuple[float, float]] = None,
                cmap    : Optional[Colormap]            = None,
                figsize : Tuple[float, float]           = (14,10)):
    """
    draw a specific map defined by parameter wmap.

    Parameters
    ----------
    aMap
        A container of maps.
        class ASectorMap:
            chi2  : DataFrame
            e0    : DataFrame
            lt    : DataFrame
            e0u   : DataFrame
            ltu   : DataFrame

    wmap
        MapType enum.
        class MapType(Enum):
            LT   = 1
            LTu  = 2
            E0   = 3
            E0u  = 4
            chi2 = 5
    atlims
        Defines the range of the map.
    cmap
        color map. For example: cmap = matplotlib.cm.viridis (defaults to seaborn)
    figsize
        Range definint the figure size.

    Returns
    -------
    Nothing
        Function produces a plot.

    """
    vmin, vmax = get_limits_(alims)
    xymap, title = which_map_(aMap, wmap, index = None)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    sns.heatmap(xymap.fillna(0) /norm, vmin=vmin, vmax=vmax, cmap=cmap, square=True)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def draw_xy_maps_ts(aMaps   : List[ASectorMap],
                    wmap    : MapType                       = MapType.LT,
                    ltlims  : Optional[Tuple[float, float]] = None,
                    ixy     : Optional[Tuple[float, float]] = None,
                    cmap    : Optional[Colormap]            = None,
                    figsize : Tuple[float, float]            = (14,10)):
    """
    draw the time series of specific map defined by parameter wmap.

    Parameters
    ----------
    aMaps
        A list of ASectorMap, one per time bin.
        class ASectorMap:
            chi2  : DataFrame
            e0    : DataFrame
            lt    : DataFrame
            e0u   : DataFrame
            ltu   : DataFrame

    wmap
        MapType enum.
        class MapType(Enum):
            LT   = 1
            LTu  = 2
            E0   = 3
            E0u  = 4
            chi2 = 5
    ltlims
        Defines the range of the map.
    ixy
        A range defining the Canvas division (plots_x, plots_y).
    cmap
        color map. For example: cmap = matplotlib.cm.viridis (defaults to seaborn)
    figsize
        Range definint the figure size.

    Returns
    -------
    Nothing
        Function produces a plot.

    """

    fig = plt.figure(figsize=figsize)
    ix, iy = get_plt_indexes_(aMaps, ixy)

    for i,  _ in enumerate(aMaps):
        ax = fig.add_subplot(ix, iy, i+1)
        xymap, title = which_map(aMaps[i], wmap, index = i)
        vmin, vmax = get_limits(ltlims)
        sns.heatmap(xymap.fillna(0), vmin=vmin, vmax=vmax, cmap=cmap, square=True)
        plt.title(title)
    plt.tight_layout()
    plt.show()


def get_limits_(alims   : Optional[Tuple[float, float]] = None)->Tuple[float,float]:
    if alims == None:
        vmin = None
        vmax = None
    else:
        vmin=alims[0]
        vmax=alims[1]
    return vmin, vmax


def get_plt_indexes_(aMaps :List[ASectorMap], ixy : Optional[Tuple[int,int]])->Tuple[int, int]:

    if ixy == None:
        if len(aMaps)%2 == 0:
            ix = len(aMaps) / 2
            iy = len(aMaps) / 2
        else:
            ix = (len(aMaps) + 1) / 2
            iy = (len(aMaps) + 1) / 2
    else:
        ix = ixy[0]
        iy = ixy[1]
    return ix, iy


def which_map_(aMap: ASectorMap,
              wmap : MapType,
              index : Optional[int] = None)->Tuple[str, DataFrame]:

    if index == None:
        title = wmap.name
    else:
        title = f'{wmap.name} : ts = {index}'

    if wmap.name == 'LT':
        ltmap = aMap.lt
    elif wmap.name == 'LTu':
        ltmap = aMap.ltu
    elif wmap.name == 'E0':
        ltmap = aMap.e0
    elif wmap.name == 'E0u':
        ltmap = aMap.E0u
    else:
        ltmap = aMap.chi2
    return ltmap, title
