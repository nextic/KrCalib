import matplotlib.pyplot as plt

from matplotlib.patches      import Circle, Wedge, Polygon
from matplotlib.collections  import PatchCollection
from matplotlib.colors       import Colormap
from matplotlib.axes         import Axes

import numpy as np
import matplotlib

from .kr_types import  KrSector, KrEvent
from . kr_types import FitType, MapType
from . kr_types import ASectorMap, TSectorMap, KrWedge
from typing    import  List, Tuple, Dict, Sequence


def rphi_sector_map(nSectors=10, rmax=200)->Tuple[Dict[int, Tuple[float, float]],
                                     Dict[int, List[Tuple[float]]]]:
    """
    Default map:

     1. 0 <  R < 20  : 0 < Phi < 360
     2. 20 < R < 40  : 0 < Phi < 180, 180 < Phi < 360
     3. 20 < R < 60  : 0 < Phi < 90, 90 < Phi < 180, 180 < Phi < 270, 270 < Phi < 360
     4. 60 < R < 80  : 0 < Phi < 45, 45 < Phi < 90, 90 < Phi < 135, 135 < Phi < 180... until 360
     5. 80 < R < 100 : 0 < Phi < 45, 45 < Phi < 90, 90 < Phi < 135, 135 < Phi < 180... until 360

     From 6 to 10:  100 < R < 200 in steps of 20 and Phi in steps of 45 degrees

    """
    PHI = {0 : [(0, 360)],
           1 : [(0,180), (180,360)],
           2 : [(i, i+90) for i in range(0, 360, 90) ]
           }

    dr = rmax / nSectors
    R = {}
    for ns in range(nSectors):
        R[ns] = (dr * ns, dr* (ns+1))

    for ns in range(3, nSectors):
        PHI[ns] = [(i, i+45) for i in range(0, 360, 45)]

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

def add_wedge_patches_to_axis(W       :  Dict[int, List[KrWedge]],
                              ax      :  Axes,
                              cmap    :  Colormap,
                              alpha   :  float,
                              rmax    :  float,
                              scale   :  float,
                              cr      :  Sequence[float],
                              clims   :  Tuple[float, float])->PatchCollection:

    for sector, krws in W.items():
        wedges = [wedge_from_sector(krw, rmax=200, scale=0.5) for krw in krws]
        colors = set_map_sequential_colors(wedges, sector, cr)
        p = PatchCollection(wedges, cmap=cmap, alpha=alpha)
        p.set_array(np.array(colors))
        ax.add_collection(p)
        p.set_clim(clims)
    return p

def draw_wedges(W       :  Dict[int, List[KrWedge]],
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


# def map_minmax(LTM, wmap, verbose):
#
#     LT = []
#     for ltl in LTM.values():
#         if wmap == MapType.LT or wmap == MapType.E0 or wmap == MapType.chi2:
#             for m in ltl:
#                 LT.append(m.value)
#         else:
#             for m in ltl:
#                 LT.append(m.uncertainty)
#
#     ltmin = min(LT)
#     ltmax = max(LT)
#
#     if verbose:
#         print(f'LT min = {ltmin}, LT max = {ltmax}')
#     return ltmin, ltmax
#
# def select_map(wmap):
#
#     if wmap == MapType.LT or wmap == MapType.LTu:
#         LTM = aSM.lt
#     elif wmap == MapType.E0 or wmap == MapType.E0u:
#         LTM = aSM.e0
#     else:
#         LTM = aSM.chi2
#
# def set_wedges(W, wmap, ax, ltmin, ltmax,
#                rmax=200, scale=0.5, cmap=matplotlib.cm.jet, alpha=0.4):
#
#     for wk in W.keys():
#         wedges = [wedge_from_sector(s, rmax=rmax, scale=scale) for s in W[wk]]
#         if wmap == MapType.LT or wmap == MapType.E0 or wmap == MapType.chi2:
#             colors = [(LTM[wk][i]).value for i in range(len(wedges)) ]
#         else:
#             colors = [(LTM[wk][i]).uncertainty for i in range(len(wedges)) ]
#         #print(colors)
#         p = PatchCollection(wedges, cmap=cmap, alpha=alpha)
#         p.set_array(np.array(colors))
#         ax.add_collection(p)
#         p.set_clim([ltmin, ltmax])
#
# def draw_map(W       : Dict[int, List[Tuple[float,float,float,float]]],
#              aSM     : ASectorMap,
#              wmap    : MapType = MapType.LT,
#              verbose : bool = True,
#              figsize=(10,8)):
#
#     fig = plt.figure(figsize=figsize) # give plots a rectangular frame
#     ax = fig.add_subplot(111)
#
#     LTM = select_map(wmap)
#
#     ltmin, ltmax = map_minmax(LTM, wmap, verbose)
#
#     for wk in W.keys():
#         wedges = [wedge_from_sector(s, rmax=200, scale=0.5) for s in W[wk]]
#         if wmap == MapType.LT or wmap == MapType.E0 or wmap == MapType.chi2:
#             colors = [(LTM[wk][i]).value for i in range(len(wedges)) ]
#         else:
#             colors = [(LTM[wk][i]).uncertainty for i in range(len(wedges)) ]
#         #print(colors)
#         p = PatchCollection(wedges, cmap=matplotlib.cm.jet, alpha=0.4)
#         p.set_array(np.array(colors))
#         ax.add_collection(p)
#         p.set_clim([ltmin, ltmax])
#     fig.colorbar(p, ax=ax)
#
#     plt.show()
#
#
# def draw_map_time_slice(W       : Dict[int, List[Tuple[float,float,float,float]]],
#                         aSM     : TSectorMap,
#                         tslice  : int,
#                         wmap    : MapType = MapType.LT,
#                         verbose : bool = True,
#                         figsize=(10,8)):
#
#     fig = plt.figure(figsize=figsize) # give plots a rectangular frame
#     ax = fig.add_subplot(111)
#
#
#     XTM = select_map(wmap)
#     if wmap == MapType.LT or wmap == MapType.LTu:
#         XTM = aSM.lt
#     elif wmap == MapType.E0 or wmap == MapType.E0u:
#         XTM = aSM.e0
#     else:
#         XTM = aSM.chi2
#
#     LTM = {}
#
#     for key in XTM.keys():
#         lts  = XTM[key]
#         NLT = [lt[tslice] for lt in lts]
#         LTM[key] = NLT
#
#
#
#     ltmin, ltmax = map_minmax(LTM, wmap, verbose)
#
#     for wk in W.keys():
#         wedges = [wedge_from_sector(s, rmax=200, scale=0.5) for s in W[wk]]
#         if wmap == MapType.LT or wmap == MapType.E0 or wmap == MapType.chi2:
#             colors = [(LTM[wk][i]).value for i in range(len(wedges)) ]
#         else:
#             colors = [(LTM[wk][i]).uncertainty for i in range(len(wedges)) ]
#         #print(colors)
#         p = PatchCollection(wedges, cmap=matplotlib.cm.jet, alpha=0.4)
#         p.set_array(np.array(colors))
#         ax.add_collection(p)
#         p.set_clim([ltmin, ltmax])
#     fig.colorbar(p, ax=ax)
#
#     plt.show()
