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

from . kr_types        import RPhiMapDef
from . kr_types        import PlotLabels
from . kr_types        import KrSector, KrEvent
from . kr_types        import FitType, MapType
from . kr_types        import FitParTS
from . kr_types        import ASectorMap, SectorMapTS, FitMapValue

from typing            import List, Tuple, Dict, Sequence, Iterable
from typing            import Optional

from numpy import sqrt
import logging
log = logging.getLogger()


def tsmap_from_fmap(fMap    : Dict[int, List[FitParTS]]) ->SectorMapTS:

    logging.debug(f' --tsmap_from_fmap')

    tmChi2  = {}
    tmE0    = {}
    tmLT    = {}
    tmE0u   = {}
    tmLTu   = {}

    for sector, fps in fMap.items():
        logging.debug(f' filling maps for sector {sector}')

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
