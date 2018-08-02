import numpy as np

from typing      import Tuple
from typing      import Dict
from typing      import List
from typing      import TypeVar

from enum        import Enum

from dataclasses import dataclass

from   invisible_cities.types.ic_types import minmax
from   invisible_cities.evm  .ic_containers  import Measurement
from   invisible_cities.evm.ic_containers import FitFunction

Number = TypeVar('Number', int, float)
Range = TypeVar('Range', None, Tuple[float])


class FitType(Enum):
    profile = 1
    unbined = 2


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
class KrRaw(Point):
    """Adds raw energy/time"""
    S2e  : np.array
    S1e  : np.array
    S2q  : np.array
    T    : np.array # time


@dataclass
class KrEvent(KrRaw):
    """Adss corrected energies"""
    Elt  : np.array
    E    : np.array
    Qlt  : np.array
    Q    : np.array


@dataclass
class  KrTimes:
    times      : List[float]
    timeStamps : List[float]
    TL         : List[Tuple[float]]


@dataclass
class FitPar:
    f     : FitFunction
    x     : np.array
    y     : np.array
    yu    : np.array


@dataclass
class HistoPar:
    var    : np.array
    nbins : int
    range : Tuple[float]


@dataclass
class HistoPar2(HistoPar):
    var2    : np.array
    nbins2 : int
    range2 : Tuple[float]


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
class MapXY:
    xs    : np.array
    ys    : np.array
    value : np.array
    error : np.array
    valid : np.array


@dataclass
class MapXYFitC(MapXY):
    fc    : List[List[FitCollection]]


@dataclass
class PlotLabels:
    x     : str
    y     : str
    title : str

#------

@dataclass
class DstEvent:
    full  : KrEvent
    fid   : KrEvent
    core  : KrEvent
    hcore : KrEvent

@dataclass
class FitCollections:
    full  : FitCollection
    fid   : FitCollection
    core  : FitCollection
    hcore : FitCollection


@dataclass
class KrLTSlices:
    Es    : np.array
    LT    : np.array
    chi2  : np.array
    valid : np.array


@dataclass
class KrLTLimits:
    Es  : minmax
    LT  : minmax
    Eu  : minmax
    LTu : minmax




@dataclass
class KrMean:
    mu    : float
    mu_u  : float


@dataclass
class KrMeanAndStd(KrMean):
    std   : float
    std_u : float


@dataclass
class KrMeanStdMinMax(KrMeanAndStd):
    min     : float
    max     : float
    min_u   : float
    max_u   : float


@dataclass
class KrMeans:
    mu    : np.array
    mu_u  : np.array


@dataclass
class KrMeansAndStds(KrMeans):
    std   : np.array
    std_u : np.array
