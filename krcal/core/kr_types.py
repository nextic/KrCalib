import numpy as np

from typing      import Tuple
from typing      import Dict
from typing      import List
from typing      import TypeVar
from typing      import Optional

from enum        import Enum
from enum        import auto

from dataclasses import dataclass
from pandas import DataFrame, Series

from invisible_cities.types.ic_types      import minmax
from invisible_cities.types.ic_types      import AutoNameEnumBase
from invisible_cities.evm  .ic_containers import Measurement
from invisible_cities.evm  .ic_containers import FitFunction

Number = TypeVar('Number', None, int, float)
Str   = TypeVar('Str', None, str)
Range = TypeVar('Range', None, Tuple[float, float])
Array = TypeVar('Array', List, np.array)
#Sel   = TypeVar('Sel',   None, np.array)  # where np.array is array of bool
#LMeasurement =  List[Measurement]
Int = TypeVar('Int', None, int)


class type_of_signal(AutoNameEnumBase):
    nS1 = auto()
    nS2 = auto()


class FitType(Enum):
    profile = 1
    unbined = 2

class MapType(Enum):
    LT   = 1
    LTu  = 2
    E0   = 3
    E0u  = 4
    chi2 = 5

@dataclass
class KrFileName:
    input_file_names : List[str]
    output_file_name : str
    map_file_name    : str
    map_file_name_ts : str
    emap_file_name   : str


@dataclass
class S1D:
    """S1 description"""
    E  : Measurement
    W  : Measurement
    H  : Measurement
    R  : Measurement # R = H/E
    T  : Measurement


@dataclass
class S2D:
    """S2 description"""
    E  : Measurement
    W  : Measurement
    Q  : Measurement
    N  : Measurement # NSipm
    X  : Measurement
    Y  : Measurement


@dataclass
class CPoint:
    """Represent a cartesian point"""
    X   : np.array
    Y   : np.array
    Z   : np.array


@dataclass
class Point(CPoint):
    """Add polar representation"""
    R   : np.array
    Phi : np.array


@dataclass
class KrEvent(Point):
    """Adds raw energy/time"""
    S2e  : Array
    S1e  : Array
    S2q  : Array
    T    : Array  # time
    DT   : Array  # time difference in seconds
    E    : Array
    Q    : Array


@dataclass
class  KrTimes:
    times      : List[float]
    timeStamps : List[float]
    TL         : List[Tuple[float]]


@dataclass
class KrBins:
    S2e  : Array
    S1e  : Array
    S2q : Array
    X    : Array
    Y    : Array
    Xsi  : Array
    Ysi  : Array
    Z    : Array
    Xc   : Array
    Yc   : Array
    T    : Array
    Xp   : Number
    Yp   : Number


@dataclass
class KrNBins:
    S2e  : Int
    S1e  : Int
    S2q  : Int
    X    : Int
    Y    : Int
    Xsi  : Int
    Ysi  : Int
    Z    : Int
    T    : Int


@dataclass
class KrRanges:
    S2e  : Range
    S1e  : Range
    S2q  : Range
    X    : Range
    Y    : Range
    Z    : Range
    T    : Range


@dataclass
class KrSector:
    rmin    : float
    rmax    : float
    phimin  : float
    phimax  : float


@dataclass
class HistoPar:
    var    : np.array
    nbins  : int
    range  : Tuple[float]


@dataclass
class HistoPar2(HistoPar):
    var2    : np.array
    nbins2 : int
    range2 : Tuple[float]


@dataclass
class ProfilePar:
    x  : np.array
    y  : np.array
    xu : np.array
    yu : np.array


@dataclass
class FitPar(ProfilePar):
    f     : FitFunction

@dataclass
class GaussPar:
    mu    : Measurement
    std   : Measurement
    amp   : Measurement


@dataclass
class FitResult:
    par  : np.array
    err  : np.array
    chi2 : float
    valid : bool


@dataclass
class FitCollection:
    fp   : FitPar
    hp   : HistoPar
    fr   : FitResult


@dataclass
class FitCollection2(FitCollection):
    fp2   : FitPar


@dataclass
class PlotLabels:
    x     : str
    y     : str
    title : str


@dataclass
class FitParTS:             # Fit parameters Time Series
    ts   : np.array          # contains the time series (integers expressing time differences)
    e0   : np.array          # e0 fitted in time series
    lt   : np.array
    c2   : np.array
    e0u  : np.array          # e0 error fitted in time series
    ltu  : np.array


@dataclass
class FitParFB:            # Fit Parameters forward-backward
    c2  : Measurement
    c2f : Measurement
    c2b : Measurement

    e0  : Measurement
    e0f : Measurement
    e0b : Measurement

    lt  : Measurement
    ltf : Measurement
    ltb : Measurement


@dataclass
class RPhiMapDef:  # defines the values in (R,Phi) to compute RPHI maps
    r   : Dict[int, Tuple[float, float]] # (rmin, rmax) in each radial sector
    phi : Dict[int, List[Tuple[float, float]]] # (phi_0, ph_1... phi_s) per radial sector


@dataclass
class SectorMapTS:  # Map in chamber sector containing time series of pars
    chi2  : Dict[int, List[np.array]]
    e0    : Dict[int, List[np.array]]
    lt    : Dict[int, List[np.array]]
    e0u   : Dict[int, List[np.array]]
    ltu   : Dict[int, List[np.array]]


@dataclass
class ASectorMap:  # Map in chamber sector containing average of pars
    chi2    : DataFrame
    e0      : DataFrame
    lt      : DataFrame
    e0u     : DataFrame
    ltu     : DataFrame
    mapinfo : Optional[Series]


@dataclass
class FitMapValue:  # A ser of values of a FitMap
    chi2  : float
    e0    : float
    lt    : float
    e0u   : float
    ltu   : float

#------


# @dataclass
# class KrLTSlices:
#     Es    : np.array
#     LT    : np.array
#     chi2  : np.array
#     valid : np.array
#
#
# @dataclass
# class KrLTLimits:
#     Es  : minmax
#     LT  : minmax
#     Eu  : minmax
#     LTu : minmax
#
#
#
#
# @dataclass
# class KrMean:
#     mu    : float
#     mu_u  : float
#
#
# @dataclass
# class KrMeanAndStd(KrMean):
#     std   : float
#     std_u : float
#
#
# @dataclass
# class KrMeanStdMinMax(KrMeanAndStd):
#     min     : float
#     max     : float
#     min_u   : float
#     max_u   : float
#
#
# @dataclass
# class KrMeans:
#     mu    : np.array
#     mu_u  : np.array
#
#
# @dataclass
# class KrMeansAndStds(KrMeans):
#     std   : np.array
#     std_u : np.array
