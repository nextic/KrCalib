import numpy as np
import matplotlib.pyplot as plt
from   invisible_cities.core.core_functions    import in_range
from   invisible_cities.evm  .ic_containers    import Measurement
from   typing                                  import Tuple, List
from . kr_types                                import Number, Range

def mean_and_std(x : np.array, range_ : Tuple[Number, Number])->Tuple[Number, Number]:
    """Computes mean and std for an array within a range"""

    mu = np.mean(x[in_range(x, *range_)])
    std = np.std(x[in_range(x, *range_)])
    return mu, std


def gaussian_experiment(nevt : Number = 1e+3,
                        mean : float  = 100,
                        std  : float  = 10)->np.array:

    Nevt  = int(nevt)
    e  = np.random.normal(mean, std, Nevt)
    return e


def gaussian_experiments(mexperiments : Number   = 1000,
                         nsample      : Number   = 1000,
                         mean         : float    = 1e+4,
                         std          : float    = 100)->List[np.array]:

    return [gaussian_experiment(nsample, mean, std) for i in range(mexperiments)]


def gaussian_experiments_variable_mean_and_std(mexperiments : Number   = 1000,
                                               nsample      : Number   = 100,
                                               mean_range   : Range    =(100, 1000),
                                               std_range    : Range    =(1, 50))->List[np.array]:
    Nevt   = int(mexperiments)
    sample = int(nsample)
    stds   = np.random.uniform(low=std_range[0], high=std_range[1], size=sample)
    means  = np.random.uniform(low=mean_range[0], high=mean_range[1], size=sample)
    return [gaussian_experiment(Nevt, mean, std) for mean in means for std in stds]


def energy_lt(z : np.array, e0: float, lt: float)->np.array:
    """Energy attenuated by lifetime"""
    e = e0 * np.exp(-z/lt)
    return e


def smear_e(e : np.array, std : float)->np.array:
    return np.array([np.random.normal(x, std) for x in e])


def energy_lt_experiment(nevt : Number   = 1e+3,
                         e0   : float = 1e+4,
                         lt   : float = 2e+3,
                         std  : float = 200,
                         zmin : float =    1,
                         zmax : float =  500)->Tuple[float, float]:

    z = np.random.uniform(low=zmin, high=zmax, size=int(nevt))
    e = energy_lt(z, e0, lt)
    es = smear_e(e, std)
    return z, es


def energy_lt_experiments(mexperiments : Number   = 1000,
                          nsample      : Number   = 1000,
                          e0           : float = 1e+4,
                          lt           : float = 2e+3,
                          std          : float = 0.02)->Tuple[np.array, np.array]:

    exps = [energy_lt_experiment(nsample, e0, lt, std) for i in range(int(mexperiments))]
    zs    = [x[0] for x in exps]
    es    = [x[1] for x in exps]
    return zs, es
