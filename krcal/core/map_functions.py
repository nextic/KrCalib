import matplotlib.pyplot as plt

from matplotlib.patches      import Circle, Wedge, Polygon
from matplotlib.collections  import PatchCollection
from matplotlib.colors       import Colormap
from matplotlib.axes         import Axes

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib

from pandas import DataFrame

from   invisible_cities.evm.ic_containers  import Measurement

from . stat_functions import  mean_and_std
from . core_functions import  NN

from . kr_types        import PlotLabels
from .kr_types         import KrSector, KrEvent
from . kr_types        import FitType, MapType
from . kr_types        import FitParTS
from . kr_types        import ASectorMap, SectorMapTS, FitMapValue

from typing            import List, Tuple, Dict, Sequence, Iterable
from typing            import Optional

from numpy import sqrt


def rphi_sector_equal_area_map(rmin : float  =  18,
                               rmax : float  = 180,
                               sphi : float =45)->Tuple[Dict[int, Tuple[float, float]],
                                     Dict[int, List[Tuple[float, float]]]]:
    # PHI = {0 : [(0, 360)],
    #        1 : [(0,180), (180,360)],
    #        2 : [(i, i+90) for i in range(0, 360, 90) ]
    #        }
    nSectors = int((rmax / rmin)**2)
    print(f'nSectors = {nSectors}')
    R = {}
    PHI = {}
    ri =[np.sqrt(i) * rmin for i in range(nSectors + 1)]
    #print(ri)

    for ns in range(nSectors):

        R[ns] = (ri[ns], ri[ns+1])
        #print(f'R[{ns}] =({ri[ns]},{ri[ns+1]})')

    for ns in range(0, nSectors):
        PHI[ns] = [(i, i+sphi) for i in range(0, 360, sphi)]

    return R, PHI


def rphi_sector_alpha_map(rmin : float  =  20,
                          rmax : float  = 200,
                          alpha: float  = 0.4,
                          sphi : float  = 5)->Tuple[Dict[int, Tuple[float, float]],
                                                    Dict[int, List[Tuple[float, float]]]]:
    def ns(alpha, rmin, rmax):
        return (rmax/rmin)**2 * (1 - alpha) + alpha

    def rn(n, alpha, rmin):
        if n == 0:
            return 0
        else:
            return rmin * sqrt((n - alpha)/(1 - alpha))

    nSectors = int(ns(alpha, rmin, rmax))
    print(f'rmin = {rmin}, rmax = {rmax}, alpha ={alpha}, nSectors = {nSectors}')

    R = {}
    PHI = {}
    ri =[rn(i, alpha, rmin)  for i in range(nSectors + 1)]
    #print(ri)

    for ns in range(nSectors):

        R[ns] = (ri[ns], ri[ns+1])
        #print(f'R[{ns}] =({ri[ns]},{ri[ns+1]})')

    for ns in range(0, nSectors):
        PHI[ns] = [(i, i+sphi) for i in range(0, 360, sphi)]

    return R, PHI

def rphi_sector_map(nSectors : int   =10,
                    rmax     : float =200,
                    sphi     : float =45)->Tuple[Dict[int, Tuple[float, float]],
                                     Dict[int, List[Tuple[float, float]]]]:
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
            #print(f'sector = {sector}')
            #print(f'w = {w}')
            T = [mean_and_std(v, range_v) for v in w]
            P = list(zip(*T))
            #print(f'T = {T}')
            #print(f'P = {P}')

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

    return ASectorMap(chi2   = pd.DataFrame.from_dict(mChi2),
                      e0    = pd.DataFrame.from_dict(mE0),
                      lt    = pd.DataFrame.from_dict(mLT),
                      e0u   = pd.DataFrame.from_dict(mE0u),
                      ltu   = pd.DataFrame.from_dict(mLTu))


def map_average(aMaps : List[ASectorMap])->ASectorMap:
    mapAV = aMaps[0]
    chi2 = mapAV.chi2
    e0   = mapAV.e0
    lt   = mapAV.lt
    e0u  = mapAV.e0u
    ltu  = mapAV.ltu

    for i in range(1, len(aMaps)):
        chi2 = chi2.add(aMaps[i].chi2)
        e0   = e0.  add(aMaps[i].e0)
        lt   = lt.  add(aMaps[i].lt)
        e0u  = e0u. add(aMaps[i].e0u)
        ltu  = ltu. add(aMaps[i].ltu)

    return ASectorMap(chi2 / len(aMaps),
                        e0 / len(aMaps),
                        lt/ len(aMaps),
                       e0u/ len(aMaps),
                       ltu/ len(aMaps))


def get_maps_from_tsmap(tsm     : Dict[int, List[float]],
                        times   : np.array,
                        erange  : Tuple[float, float] = (2000, 14000),
                        ltrange : Tuple[float, float] = (500,5000),
                        c2range : Tuple[float, float] = (0,3),
                        debug   : bool                = False)->List[ASectorMap]:
    """Extracts maps for each time tranch, regularizes the maps and sets relative errors"""

    aMaps = []
    for i, _ in enumerate(times):
        am = amap_from_tsmap(tsm,
                             ts = i,
                             range_e     = erange,
                             range_chi2  = c2range,
                             range_lt    = ltrange)

        rmap = regularize_maps(am, erange=erange, ltrange=ltrange, debug=debug)
        asm = relative_errors(rmap)
        aMaps.append(asm)
    return aMaps


def amap_valid_mask(amap : ASectorMap)->ASectorMap:

    def valid_mask(df):
        vMask ={}
        for i in df.columns:
            vMask[i] =~np.isnan(df[i])
        return pd.DataFrame.from_dict(vMask)

    return ASectorMap(chi2  = valid_mask(amap.chi2),
                      e0    = valid_mask(amap.e0),
                      lt    = valid_mask(amap.lt),
                      e0u   = valid_mask(amap.e0u),
                      ltu   = valid_mask(amap.ltu))


def amap_average(amap : ASectorMap)->FitMapValue:
    return ASectorMap(chi2  = amap.chi2.mean().mean(),
                      e0    = amap.e0.mean().mean(),
                      lt    = amap.lt.mean().mean(),
                      e0u   = amap.e0u.mean().mean(),
                      ltu   = amap.ltu.mean().mean())


def amap_max(amap : ASectorMap)->FitMapValue:
    return ASectorMap(chi2  = amap.chi2.max().max(),
                      e0    = amap.e0.max().max(),
                      lt    = amap.lt.max().max(),
                      e0u   = amap.e0u.max().max(),
                      ltu   = amap.ltu.max().max())


def amap_min(amap : ASectorMap)->FitMapValue:
    return ASectorMap(chi2  = amap.chi2.min().min(),
                      e0    = amap.e0.min().min(),
                      lt    = amap.lt.min().min(),
                      e0u   = amap.e0u.min().min(),
                      ltu   = amap.ltu.min().min())


def amap_replace_nan_by_mean(amap : ASectorMap, amMean : FitMapValue)->ASectorMap:

    return ASectorMap(chi2  = amap.chi2.copy().fillna(amMean.chi2),
                      e0    = amap.e0.copy().fillna(amMean.e0),
                      lt    = amap.lt.copy().fillna(amMean.lt),
                      e0u   = amap.e0u.copy().fillna(amMean.e0u),
                      ltu   = amap.ltu.copy().fillna(amMean.ltu))


def amap_replace_nan_by_zero(amap : ASectorMap)->ASectorMap:
    return ASectorMap(chi2  = amap.chi2.copy().fillna(0),
                      e0    = amap.e0.copy().fillna(0),
                      lt    = amap.lt.copy().fillna(0),
                      e0u   = amap.e0u.copy().fillna(0),
                      ltu   = amap.ltu.copy().fillna(0))



def amap_valid_fraction(vmask: ASectorMap)->FitMapValue:

    def count_valid(df):
        C = []
        for i in df.columns:
            C.append(np.count_nonzero(df[i]))
        return np.sum(C) /df.size

    return FitMapValue(chi2  = count_valid(vmask.chi2),
                      e0    = count_valid(vmask.e0),
                      lt    = count_valid(vmask.lt),
                      e0u   = count_valid(vmask.e0u),
                      ltu   = count_valid(vmask.ltu))

    FitMapValue


def relative_errors(am : ASectorMap)->ASectorMap:
    return ASectorMap(chi2  = am.chi2,
                      e0    = am.e0,
                      lt    = am.lt,
                      e0u   = 100 * am.e0u / am.e0,
                      ltu   = 100 * am.ltu / am.lt)


def regularize_maps(amap    : ASectorMap,
                    erange  : Tuple[float, float] = (2000, 14000),
                    ltrange : Tuple[float, float] = (500,5000),
                    debug   : bool = False)->ASectorMap:

    OL   = find_outliers(amap.e0, xr=erange, debug=debug)
    me0  = set_outliers_to_nan(amap.e0, OL)
    me0u = set_outliers_to_nan(amap.e0u, OL)
    OL   = find_outliers(amap.lt, xr=ltrange, debug=debug)
    mlt  = set_outliers_to_nan(amap.lt, OL)
    mltu = set_outliers_to_nan(amap.ltu, OL)
    return ASectorMap(chi2  = amap.chi2,
                      e0    =       me0,
                      lt    =       mlt,
                      e0u   =       me0u,
                      ltu   =       mltu)


def set_outliers_to_nan(dfmap : DataFrame, OL : Dict[int, List[int]])->DataFrame:
    newmap = dfmap.copy()
    for i, lst in OL.items():
        for j in lst:
            newmap[i][j] = np.nan
    return newmap


def find_outliers(dfmap : DataFrame,
                  xr    : Tuple[float,float],
                  debug : bool  = False)->Dict[int, List[int]]:
    OL = {}
    v = (xr[1] + xr[0]) / 2
    if debug:
        print(f' set nans to average value of interval = {v}')
    newmap = (dfmap.copy()).fillna(v)
    for i in newmap.columns:
        ltc = newmap[i]
        gltc = ltc.between(*xr)
        lst = list(gltc[gltc==False].index)
        if len(lst) > 0:
            OL[i] = lst
            if debug:
                print(f'column {i}')
            for il in lst:
                if debug:
                    print(f'outlier found, index = {il}, value ={ltc[il]}')
    return OL


def energy_map(KRES : Dict[int, List[KrEvent]])->DataFrame:

    wedges =[len(kre) for kre in KRES.values() ]  # number of wedges per sector
    eMap = {}

    for sector in KRES.keys():
        eMap[sector] = [np.mean(KRES[sector][i].E) for i in range(wedges[sector])]
    return pd.DataFrame.from_dict(eMap)


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
        e0M = LTM.max().max()
        e0m = LTM.min().min()
        return e0m, e0M


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


def draw_map(W       : Dict[int, List[KrSector]],
             aMap    : DataFrame,
             alims   : Optional[Tuple[float, float]] = None,
             title   : str                           = 'E',
             cmap    :  Colormap                      = matplotlib.cm.viridis,
             alpha   : float                          = 1.0,  # level of transparency
             rmax    : float                          = 200,  # the largest radius
             scale   : float                          = 0.5,  # needed to fit the map
             figsize : Tuple[float, float]            = (14,10)):


    fig = plt.figure(figsize=figsize) # give plots a rectangular frame

    ax = fig.add_subplot(1,1,1)
    if alims == None:
        e0M = aMap.max().max()
        e0m = aMap.min().min()
    else:
        e0m, e0M = alims[0], alims[1]
    p = add_map_values_to_axis(W, aMap, ax, cmap, alpha, rmax, scale, clims=(e0m, e0M))
    fig.colorbar(p, ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def draw_maps_ts(W       : Dict[int, List[KrSector]],
                 aMaps   : List[ASectorMap],
                 wmap    : MapType                       = MapType.LT,
                 ltlims  : Optional[Tuple[float, float]] = None,
                 ixy     : Optional[Tuple[float, float]] = None,
                 cmap    :  Colormap                      = matplotlib.cm.viridis,
                 alpha   : float                          = 1.0,  # level of transparency
                 rmax    : float                          = 200,  # the largest radius
                 scale   : float                          = 0.5,  # needed to fit the map
                 figsize : Tuple[float, float]            = (14,10)):

    fig = plt.figure(figsize=figsize)
    if ixy == None:
        if len(aMaps)%2 == 0:
            ix = len(aMaps) / 2
            iy = len(aMaps) / 2
        else:
            ix = len(aMaps) + 1 / 2
            iy = len(aMaps) + 1 / 2
    else:
        ix = ixy[0]
        iy = ixy[1]
    for i, aMap in enumerate(aMaps):
        ax = fig.add_subplot(ix,iy,i+1)
        ltmap, title = which_map(aMap, wmap, index = i)

        # if wmap == MapType.LT:
        #     title = f'LT : ts = {i}'
        #     ltmap = aMap.lt
        # elif wmap == MapType.LTu:
        #     title = f'LTu : ts = {i}'
        #     ltmap = aMap.ltu
        # elif wmap == MapType.E0:
        #     title = f'E0 : ts = {i}'
        #     ltmap = aMap.e0
        # elif wmap == MapType.E0u:
        #     title = f'E0u : ts = {i}'
        #     ltmap = aMap.E0u
        # else:
        #     title = f'Chi2 : ts = {i}'
        #     ltmap = aMap.chi2

        if ltlims == None:
            e0M = ltmap.max().max()
            e0m = ltmap.min().min()
        else:
            e0m, e0M = ltlims[0], ltlims[1]
        p = add_map_values_to_axis(W, ltmap, ax, cmap, alpha, rmax, scale, clims=(e0m, e0M))
        fig.colorbar(p, ax=ax)
        plt.title(title)
    plt.tight_layout()
    plt.show()


def draw_xy_maps(aMap    : ASectorMap,
                e0lims   : Optional[Tuple[float, float]] = None,
                ltlims   : Optional[Tuple[float, float]] = None,
                eulims   : Optional[Tuple[float, float]] = None,
                lulims   : Optional[Tuple[float, float]] = None,
                cmap    :  Optional[Colormap]            = None,
                figsize : Tuple[float, float]            = (14,10)):

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
                cmap    :  Colormap                      = matplotlib.cm.viridis,
                figsize : Tuple[float, float]            = (14,10)):

    vmin, vmax = get_limits(alims)
    ltmap, title = which_map(aMap, wmap, index = None)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    sns.heatmap(ltmap.fillna(0) /norm, vmin=vmin, vmax=vmax, cmap=cmap, square=True)

    plt.title(title)
    plt.tight_layout()
    plt.show()

def draw_xy_maps_ts(aMaps   : List[ASectorMap],
                    wmap    : MapType                       = MapType.LT,
                    ltlims  : Optional[Tuple[float, float]] = None,
                    ixy     : Optional[Tuple[float, float]] = None,
                    cmap    : Optional[Colormap]            = None,
                    figsize : Tuple[float, float]            = (14,10)):


    fig = plt.figure(figsize=figsize)
    ix, iy = get_plt_indexes(aMaps, ixy)

    for i,  _ in enumerate(aMaps):
        ax = fig.add_subplot(ix, iy, i+1)
        ltmap, title = which_map(aMaps[i], wmap, index = i)
        vmin, vmax = get_limits(ltlims)
        sns.heatmap(ltmap.fillna(0), vmin=vmin, vmax=vmax, cmap=cmap, square=True)
        plt.title(title)
    plt.tight_layout()
    plt.show()


def get_limits(alims   : Optional[Tuple[float, float]] = None)->Tuple[float,float]:
    if alims == None:
        vmin = None
        vmax = None
    else:
        vmin=alims[0]
        vmax=alims[1]
    return vmin, vmax


def get_plt_indexes(aMaps :List[ASectorMap], ixy : Optional[Tuple[int,int]])->Tuple[int, int]:

    if ixy == None:
        if len(aMaps)%2 == 0:
            ix = len(aMaps) / 2
            iy = len(aMaps) / 2
        else:
            ix = len(aMaps) + 1 / 2
            iy = len(aMaps) + 1 / 2
    else:
        ix = ixy[0]
        iy = ixy[1]
    return ix, iy


def which_map(aMap: ASectorMap,
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
