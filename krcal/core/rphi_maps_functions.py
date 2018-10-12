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

from . kr_types        import RPhiMapDef
from . kr_types        import KrSector
from . kr_types        import KrEvent
from . kr_types        import MapType
from . kr_types        import ASectorMap

from typing            import List, Tuple, Dict, Sequence, Iterable
from typing            import Optional

from numpy import sqrt
import logging
log = logging.getLogger()


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

    for ns in range(nSectors):

        R[ns] = (ri[ns], ri[ns+1])

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


def rphi_sector_map_def(nSectors : int   =10,
                        rmax     : float =200,
                        sphi     : float =45)->RPhiMapDef:
    """Returns a RPhiMapDef, which defines the values in (R,Phi) to compute RPHI maps

    class RPhiMapDef:
        r   : Dict[int, Tuple[float, float] -> (rmin, rmax) in each radial sector
        phi : Dict[int, List[Tuple[float, float] -> (phi_0, ph_1... phi_s) per radial sector

    """

    PHI = {}

    dr = rmax / nSectors
    R = {}
    for ns in range(nSectors):
        ri = dr * ns
        rs = dr* (ns+1)
        R[ns] = (ri, rs)

    for ns in range(0, nSectors):
        PHI[ns] = [(i, i+sphi) for i in range(0, 360, sphi)]

    return RPhiMapDef(R, PHI)


def define_rphi_sectors(rpmf : RPhiMapDef)-> Dict[int, List[KrSector]]:
    """For each radial sector (key of the Dict[int, List[KrSector]])
       returns a list of KrSector
       class KrSector:
           rmin    : float
           rmax    : float
           phimin  : float
           phimax  : float
    """

    def rps_sector(sector_number : int  ,
                   rmin          : float,
                   rmax          : float,
                   Phid          : List[Tuple[float]])->List[KrSector]:

        logging.debug('--rps_sector():')

        logging.debug(f'Sector number = {sector_number}, Rmin = {rmin}, Rmax = {rmax}')
        logging.debug(f'Number of Phi wedges = {len(Phid)}')
        logging.debug(f'Phi Wedges = {Phid}')

        rps   =  [KrSector(rmin = rmin,
                           rmax = rmax,
                           phimin=phi[0], phimax=phi[1]) for phi in Phid]
        return rps

    logging.debug('--define_rphi_sectors():')
    RPS = {}
    R   = rpmf.r
    PHI = rpmf.phi
    assert len(R.keys()) == len(PHI.keys())

    for i, r in R.items():
        RPS[i] = rps_sector(sector_number = i,
                            rmin = r[0],
                            rmax = r[1],
                            Phid = PHI[i])

    return RPS


def wedge_from_sector_(s     : KrSector,
                      rmax  : float =200,
                      scale : float =0.1) ->Wedge:
    w =  Wedge((0.5, 0.5), scale*s.rmax/rmax, s.phimin, s.phimax,
               width=scale*(s.rmax - s.rmin)/rmax)
    return w


def set_map_sequential_colors(wedges : Wedge,
                              sector : int,
                              cr     :  Sequence[float]):

    return [i + cr[sector] for i in range(len(wedges)) ]


def draw_wedges(W       :  Dict[int, List[KrSector]],
                cmap    :  Colormap                = matplotlib.cm.viridis,
                alpha   :  float                   = 0.4,  # level of transparency
                rmax    :  float                   = 200,  # the largest radius
                scale   :  float                   = 0.5,  # needed to fit the map
                figsize :  Tuple[float, float]     =(10,8),
                cr      :  Sequence[float]         =(0,5,10,20,30,40,50,60,70,80),
                clims   :  Tuple[float, float]     = (0, 100)):

    def add_wedge_patches_to_axis(W       :  Dict[int, List[KrSector]],
                                  ax      :  Axes,
                                  cmap    :  Colormap,
                                  alpha   :  float,
                                  rmax    :  float,
                                  scale   :  float,
                                  cr      :  Sequence[float],
                                  clims   :  Tuple[float, float])->PatchCollection:

        for sector, krws in W.items():
            wedges = [wedge_from_sector_(krw, rmax=rmax, scale=scale) for krw in krws]
            colors = set_map_sequential_colors(wedges, sector, cr)
            p = PatchCollection(wedges, cmap=cmap, alpha=alpha)
            p.set_array(np.array(colors))
            ax.add_collection(p)
            p.set_clim(clims)
        return p

    fig = plt.figure(figsize=figsize) # give plots a rectangular frame
    ax = fig.add_subplot(111)

    p = add_wedge_patches_to_axis(W, ax, cmap, alpha, rmax, scale, cr, clims)
    fig.colorbar(p, ax=ax)

    plt.show()


def energy_map_rphi(KRES : Dict[int, List[KrEvent]])->DataFrame:

    wedges =[len(kre) for kre in KRES.values() ]  # number of wedges per sector
    eMap = {}

    for sector in KRES.keys():
        eMap[sector] = [np.mean(KRES[sector][i].E) for i in range(wedges[sector])]
    return pd.DataFrame.from_dict(eMap)


def add_map_values_to_axis_(W       :  Dict[int, List[KrSector]],
                           M       :  Dict[int, List[float]],
                           ax      :  Axes,
                           cmap    :  Colormap,
                           alpha   :  float,
                           rmax    :  float,
                           scale   :  float,
                           clims   :  Tuple[float, float])->PatchCollection:

    for sector, krws in W.items():
        wedges = [wedge_from_sector_(krw, rmax=rmax, scale=scale) for krw in krws]
        colors = [M[sector][i] for i in range(len(wedges)) ]
        #print(colors)
        p = PatchCollection(wedges, cmap=cmap, alpha=alpha)
        p.set_array(np.array(colors))
        ax.add_collection(p)
        p.set_clim(clims)
    return p


def draw_rphi_maps(W       : Dict[int, List[KrSector]],
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
    p = add_map_values_to_axis_(W, aMap.e0, ax, cmap, alpha, rmax, scale, clims=(e0m, e0M))
    fig.colorbar(p, ax=ax)
    plt.title('e0')

    ax = fig.add_subplot(2,2,2)
    if eulims == None:
        e0um, e0uM = map_minmax(aMap.e0u)
    else:
        e0um, e0uM = eulims[0], eulims[1]
    p = add_map_values_to_axis_(W, aMap.e0u, ax, cmap, alpha, rmax, scale, clims=(e0um, e0uM))
    fig.colorbar(p, ax=ax)
    plt.title('e0u')

    ax = fig.add_subplot(2,2,3)
    if ltlims == None:
        ltm, ltM = map_minmax(aMap.lt)
    else:
        ltm, ltM = ltlims[0], ltlims[1]

    p = add_map_values_to_axis_(W, aMap.lt, ax, cmap, alpha, rmax, scale, clims=(ltm, ltM))
    fig.colorbar(p, ax=ax)
    plt.title('LT')

    ax = fig.add_subplot(2,2,4)
    if lulims == None:
        ltum, ltuM = map_minmax(aMap.ltu)
    else:
        ltum, ltuM = lulims[0], lulims[1]
    p = add_map_values_to_axis_(W, aMap.ltu, ax, cmap, alpha, rmax, scale, clims=(ltum, ltuM))
    fig.colorbar(p, ax=ax)
    plt.title('LTu')
    plt.show()


def draw_map_rphi(W       : Dict[int, List[KrSector]],
                  aMap    : DataFrame,
                  alims   : Optional[Tuple[float, float]] = None,
                  title   : str                           = 'E',
                  cmap    :  Colormap                     = matplotlib.cm.viridis,
                  alpha   : float                         = 1.0,  # level of transparency
                  rmax    : float                         = 200,  # the largest radius
                  scale   : float                         = 0.5,  # needed to fit the map
                  figsize : Tuple[float, float]           = (14,10)):


    fig = plt.figure(figsize=figsize) # give plots a rectangular frame

    ax = fig.add_subplot(1,1,1)
    if alims == None:
        e0M = aMap.max().max()
        e0m = aMap.min().min()
    else:
        e0m, e0M = alims[0], alims[1]
    p = add_map_values_to_axis_(W, aMap, ax, cmap, alpha, rmax, scale, clims=(e0m, e0M))
    fig.colorbar(p, ax=ax)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def draw_maps_rphi_ts(W       : Dict[int, List[KrSector]],
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

        if ltlims == None:
            e0M = ltmap.max().max()
            e0m = ltmap.min().min()
        else:
            e0m, e0M = ltlims[0], ltlims[1]
        p = add_map_values_to_axis_(W, ltmap, ax, cmap, alpha, rmax, scale, clims=(e0m, e0M))
        fig.colorbar(p, ax=ax)
        plt.title(title)
    plt.tight_layout()
    plt.show()
