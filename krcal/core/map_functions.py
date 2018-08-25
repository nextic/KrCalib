import matplotlib.pyplot as plt

from matplotlib.patches      import Circle, Wedge, Polygon
from matplotlib.collections  import PatchCollection
from matplotlib.colors       import Colormap
from matplotlib.axes         import Axes

import numpy as np
import matplotlib

from pandas import DataFrame

from   invisible_cities.evm.ic_containers  import Measurement

from . stat_functions import  mean_and_std

from . kr_types        import PlotLabels
from .kr_types         import KrSector, KrEvent
from . kr_types        import FitType, MapType
from . kr_types        import FitParTS
from . kr_types        import ASectorMap, SectorMapTS
from typing            import List, Tuple, Dict, Sequence, Iterable
from typing            import Optional


def rphi_sector_map(nSectors : int   =10,
                    rmax     : float =200,
                    sphi     : float =45)->Tuple[Dict[int, Tuple[float, float]],
                                     Dict[int, List[Tuple[float]]]]:
    # PHI = {0 : [(0, 360)],
    #        1 : [(0,180), (180,360)],
    #        2 : [(i, i+90) for i in range(0, 360, 90) ]
    #        }


    PHI = {}

    dr = rmax / nSectors
    R = {}
    for ns in range(nSectors):
        ri = dr * ns
        rs = dr* (ns+1)
        R[ns] = (ri, rs)
        #print(f'R[{ns}] =({ri},{rs})')

    for ns in range(0, nSectors):
        PHI[ns] = [(i, i+sphi) for i in range(0, 360, sphi)]

    return R, PHI

def rphi_sector_map2(ns1  : int   =10,
                     ns2  : int   =50,
                     ns3  : int   =40,
                     r1   : float =50,
                     r2   : float =150,
                     rmax : float =200,
                     sphi : float =15)->Tuple[Dict[int, Tuple[float, float]],
                                              Dict[int, List[Tuple[float]]], int]:

    PHI = {}
    R = {}

    dr = r1 / ns1
    r0 = 0
    for ns in range(ns1):
        ri = r0 + dr * ns
        rs = r0 + dr* (ns+1)
        R[ns] = (ri, rs)
        print(f'R[{ns}] =({ri},{rs})')

    dr = (r2 -r1) / ns2
    r0 = rs - dr * ns1
    for ns in range(ns1, ns1+ns2):
        ri = r0 + dr * ns
        rs = r0 + dr* (ns+1)
        R[ns] = (ri, rs)
        print(f'R[{ns}] =({ri},{rs})')

    dr = (rmax -r2) / ns3
    r0 = rs - dr * (ns1 + ns2)
    for ns in range(ns1+ns2, ns1 + ns2 + ns3):
        ri = r0 + dr * ns
        rs = r0 + dr* (ns+1)
        R[ns] = (ri, rs)
        print(f'R[{ns}] =({ri},{rs})')

    nSectors = ns1 + ns2 + ns3

    for ns in range(0, nSectors):
        PHI[ns] = [(i, i+sphi) for i in range(0, 360, sphi)]

    return R, PHI, nSectors


def define_rphi_sectors(R       : Dict[int,  Tuple[float, float]],
                        PHI     : Dict[int,  List[Tuple[float]]],
                        verbose : bool  = False)-> Dict[int, List[KrSector]]:
    """ns defines the index of dicts where wedge division becomes regular"""

    def rps_sector(sector_number : int  ,
                   rmin          : float,
                   rmax          : float,
                   Phid          : List[Tuple[float]],
                   verbose       : bool  = True)->List[KrSector]:

        if verbose:
            print(f'Sector number = {sector_number}, Rmin = {rmin}, Rmax = {rmax}')
            print(f'Number of Phi wedges = {len(Phid)}')
            print(f'Phi Wedges = {Phid}')

        rps   =  [KrSector(rmin = rmin,
                           rmax = rmax,
                           phimin=phi[0], phimax=phi[1]) for phi in Phid]
        return rps

    RPS = {}

    assert len(R.keys()) == len(PHI.keys())

    for i, r in R.items():
        RPS[i] = rps_sector(sector_number = i,
                            rmin = r[0],
                            rmax = r[1],
                            Phid = PHI[i],
                            verbose = verbose)

    return RPS


def wedge_from_sector(s     : KrSector,
                      rmax  : float =200,
                      scale : float =0.1) ->Wedge:
    w =  Wedge((0.5, 0.5), scale*s.rmax/rmax, s.phimin, s.phimax, width=scale*(s.rmax - s.rmin)/rmax)
    return w


def set_map_sequential_colors(wedges : Wedge,
                              sector : int,
                              cr     :  Sequence[float]):

    return [i + cr[sector] for i in range(len(wedges)) ]


def add_wedge_patches_to_axis(W       :  Dict[int, List[KrSector]],
                              ax      :  Axes,
                              cmap    :  Colormap,
                              alpha   :  float,
                              rmax    :  float,
                              scale   :  float,
                              cr      :  Sequence[float],
                              clims   :  Tuple[float, float])->PatchCollection:

    for sector, krws in W.items():
        wedges = [wedge_from_sector(krw, rmax=rmax, scale=scale) for krw in krws]
        colors = set_map_sequential_colors(wedges, sector, cr)
        p = PatchCollection(wedges, cmap=cmap, alpha=alpha)
        p.set_array(np.array(colors))
        ax.add_collection(p)
        p.set_clim(clims)
    return p


def draw_wedges(W       :  Dict[int, List[KrSector]],
                cmap    :  Colormap                = matplotlib.cm.viridis,
                alpha   :  float                   = 0.4,  # level of transparency
                rmax    :  float                   = 200,  # the largest radius
                scale   :  float                   = 0.5,  # needed to fit the map
                figsize :  Tuple[float, float]     =(10,8),
                cr      :  Sequence[float]         =(0,5,10,20,30,40,50,60,70,80),
                clims   :  Tuple[float, float]     = (0, 100)):

    fig = plt.figure(figsize=figsize) # give plots a rectangular frame
    ax = fig.add_subplot(111)

    p = add_wedge_patches_to_axis(W, ax, cmap, alpha, rmax, scale, cr, clims)
    fig.colorbar(p, ax=ax)

    plt.show()


def tsmap_from_fmap(fMap    : Dict[int, List[FitParTS]],
                    verbose : bool = False) ->SectorMapTS:

    tmChi2  = {}
    tmE0    = {}
    tmLT    = {}
    tmE0u   = {}
    tmLTu   = {}

    for sector, fps in fMap.items():

        if verbose:
            print(f' filling maps for sector {sector}')
            print(f' number of wedges in sector {len(fps)}')


        tCHI2 = [fp.c2 for fp in fps]
        tE0   = [fp.e0 for fp in fps]
        tLT   = [fp.lt for fp in fps]
        tE0u  = [fp.e0u for fp in fps]
        tLTu  = [fp.ltu for fp in fps]

        tmChi2[sector]  = tCHI2
        tmE0  [sector]  = tE0
        tmLT  [sector]  = tLT
        tmE0u [sector]  = tE0u
        tmLTu [sector]  = tLTu

    return SectorMapTS(chi2  = tmChi2,
                       e0    = tmE0,
                       lt    = tmLT,
                       e0u   = tmE0u,
                       ltu   = tmLTu)


def amap_from_tsmap(tsMap      : SectorMapTS,
                    ts         : int  = 0,       # if negative take the average
                    range_e    : Tuple[float, float] = (5000, 13000),
                    range_chi2 : Tuple[float, float] = (0,3),
                    range_lt   : Tuple[float, float] = (1800, 3000)) ->ASectorMap:

    def fill_map_ts(tsm : Dict[int, List[float]], ts : int):
        M = {}
        for sector, w in tsm.items():
            M[sector] = [v[ts] for v in w]

        return M

    def fill_maps_av(tsm : Dict[int, List[float]], range_v : Tuple[float, float]):
        M  = {}
        Mu = {}
        for sector, w in tsm.items():
            T = [mean_and_std(v, range_v) for v in w]
            P = list(zip(*T))
            #print(P[0])
            M[sector] = P[0]
            Mu[sector] = P[1]

        return M, Mu

    if ts >=0:
        mChi2  = fill_map_ts(tsMap.chi2, ts)
        mE0    = fill_map_ts(tsMap.e0, ts)
        mLT    = fill_map_ts(tsMap.lt, ts)
        mE0u   = fill_map_ts(tsMap.e0u, ts)
        mLTu   = fill_map_ts(tsMap.ltu, ts)
    else:
        mChi2, _   = fill_maps_av(tsMap.chi2, range_chi2)
        mE0, mE0u  = fill_maps_av(tsMap.e0, range_e)
        mLT, mLTu  = fill_maps_av(tsMap.lt, range_lt)

    return ASectorMap(chi2   = mChi2,
                      e0    = mE0,
                      lt    = mLT,
                      e0u   = mE0u,
                      ltu   = mLTu)


def relative_errors(am : ASectorMap)->ASectorMap:
    mC2  = {}
    mE0  = {}
    mLT  = {}
    mE0u = {}
    mLTu = {}


    for i, L in am.chi2.items():
        mC2[i] = L
    for i, L in am.e0.items():
        mE0[i] = L
    for i, L in am.lt.items():
        mLT[i] = L
    for i, L in am.e0u.items():
        Le0 = am.e0[i]
        Le0u =am.e0u[i]
        mE0u[i] = [100* y/x for x,y in zip(Le0, Le0u)]
    for i, L in am.ltu.items():
        Llt = am.lt[i]
        Lltu =am.ltu[i]
        mLTu[i] = [100* y/x if x > 0 else -1 for x,y in zip(Llt, Lltu) ]

    return ASectorMap(chi2  = mC2,
                      e0    = mE0,
                      lt    = mLT,
                      e0u   = mE0u,
                      ltu   = mLTu)

def add_map_values_to_axis(W       :  Dict[int, List[KrSector]],
                           M       :  Dict[int, List[float]],
                           ax      :  Axes,
                           cmap    :  Colormap,
                           alpha   :  float,
                           rmax    :  float,
                           scale   :  float,
                           clims   :  Tuple[float, float])->PatchCollection:

    for sector, krws in W.items():
        wedges = [wedge_from_sector(krw, rmax=rmax, scale=scale) for krw in krws]
        colors = [M[sector][i] for i in range(len(wedges)) ]
        #print(colors)
        p = PatchCollection(wedges, cmap=cmap, alpha=alpha)
        p.set_array(np.array(colors))
        ax.add_collection(p)
        p.set_clim(clims)
    return p

def draw_maps(W       : Dict[int, List[KrSector]],
              aMap    : ASectorMap,
              e0lims   : Optional[Tuple[float, float]] = None,
              ltlims   : Optional[Tuple[float, float]] = None,
              eulims   : Optional[Tuple[float, float]] = None,
              lulims   : Optional[Tuple[float, float]] = None,
              cmap    :  Colormap                      = matplotlib.cm.viridis,
              alpha   : float                          = 1.0,  # level of transparency
              rmax    : float                          = 200,  # the largest radius
              scale   : float                          = 0.5,  # needed to fit the map
              figsize : Tuple[float, float]            = (14,10)):

    def map_minmax(LTM):
        LT = []
        for l in LTM.values():
            for v in l:
                LT.append(v)

        return min(LT), max(LT)


    fig = plt.figure(figsize=figsize) # give plots a rectangular frame

    ax = fig.add_subplot(2,2,1)
    if e0lims == None:
        e0m, e0M = map_minmax(aMap.e0)
    else:
        e0m, e0M = e0lims[0], e0lims[1]
    p = add_map_values_to_axis(W, aMap.e0, ax, cmap, alpha, rmax, scale, clims=(e0m, e0M))
    fig.colorbar(p, ax=ax)
    plt.title('e0')

    ax = fig.add_subplot(2,2,2)
    if eulims == None:
        e0um, e0uM = map_minmax(aMap.e0u)
    else:
        e0um, e0uM = eulims[0], eulims[1]
    p = add_map_values_to_axis(W, aMap.e0u, ax, cmap, alpha, rmax, scale, clims=(e0um, e0uM))
    fig.colorbar(p, ax=ax)
    plt.title('e0u')

    ax = fig.add_subplot(2,2,3)
    if ltlims == None:
        ltm, ltM = map_minmax(aMap.lt)
    else:
        ltm, ltM = ltlims[0], ltlims[1]

    p = add_map_values_to_axis(W, aMap.lt, ax, cmap, alpha, rmax, scale, clims=(ltm, ltM))
    fig.colorbar(p, ax=ax)
    plt.title('LT')

    ax = fig.add_subplot(2,2,4)

    if lulims == None:
        ltum, ltuM = map_minmax(aMap.ltu)
    else:
        ltum, ltuM = lulims[0], lulims[1]
    p = add_map_values_to_axis(W, aMap.ltu, ax, cmap, alpha, rmax, scale, clims=(ltum, ltuM))
    fig.colorbar(p, ax=ax)
    plt.title('LTu')
    plt.show()


def draw_energy_map(W       : Dict[int, List[KrSector]],
                    eMap    : Dict[int, List[float]],
                    elims   : Optional[Tuple[float, float]] = None,
                    cmap    :  Colormap                      = matplotlib.cm.viridis,
                    alpha   : float                          = 1.0,  # level of transparency
                    rmax    : float                          = 200,  # the largest radius
                    scale   : float                          = 0.5,  # needed to fit the map
                    figsize : Tuple[float, float]            = (14,10)):

    def map_minmax(LTM):
        LT = []
        for l in LTM.values():
            for v in l:
                LT.append(v)

        return min(LT), max(LT)


    fig = plt.figure(figsize=figsize) # give plots a rectangular frame

    ax = fig.add_subplot(1,1,1)
    if elims == None:
        e0m, e0M = map_minmax(eMap)
    else:
        e0m, e0M = elims[0], elims[1]
    p = add_map_values_to_axis(W, eMap, ax, cmap, alpha, rmax, scale, clims=(e0m, e0M))
    fig.colorbar(p, ax=ax)
    plt.title('E')

    plt.show()


def draw_energy_map_df(W       : Dict[int, List[KrSector]],
                       eMap    : DataFrame,
                       elims   : Optional[Tuple[float, float]] = None,
                       cmap    :  Colormap                      = matplotlib.cm.viridis,
                       alpha   : float                          = 1.0,  # level of transparency
                       rmax    : float                          = 200,  # the largest radius
                       scale   : float                          = 0.5,  # needed to fit the map
                       figsize : Tuple[float, float]            = (14,10)):


    fig = plt.figure(figsize=figsize) # give plots a rectangular frame

    ax = fig.add_subplot(1,1,1)
    if elims == None:
        e0M = eMap.max().max()
        e0m = eMap.min().min()
    else:
        e0m, e0M = elims[0], elims[1]
    p = add_map_values_to_axis(W, eMap, ax, cmap, alpha, rmax, scale, clims=(e0m, e0M))
    fig.colorbar(p, ax=ax)
    plt.title('E')

    plt.show()


def add_map_values_to_axis_df(W       :  Dict[int, List[KrSector]],
                              M       :  DataFrame,
                              ax      :  Axes,
                              cmap    :  Colormap,
                              alpha   :  float,
                              rmax    :  float,
                              scale   :  float,
                              clims   :  Tuple[float, float])->PatchCollection:

    for sector, krws in W.items():
        wedges = [wedge_from_sector(krw, rmax=rmax, scale=scale) for krw in krws]
        colors = [M[sector][i] for i in range(len(wedges)) ]
        #print(colors)
        p = PatchCollection(wedges, cmap=cmap, alpha=alpha)
        p.set_array(np.array(colors))
        ax.add_collection(p)
        p.set_clim(clims)
    return p
