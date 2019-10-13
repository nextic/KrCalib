import numpy as np
from typing import Tuple
from typing import List

from . fit_lt_functions    import fit_lifetime

from . kr_types import FitType, Number, FitCollection2, Range

def energy_lt(z : np.array, e0: float, lt: float)->np.array:
    """Energy attenuated by lifetime"""
    e = e0 * np.exp(-z/lt)
    return e


def smear_e(e : np.array, std : float)->np.array:
    return np.array([np.random.normal(x, std) for x in e])


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
    exps   = [gaussian_experiment(Nevt, mean, std) for mean in means for std in stds]
    return means, stds, exps



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

#experiments
def fit_lifetime_experiments(zs      : np.array,
                             es      : np.array,
                             nbins_z : int      ,
                             nbins_e : int      ,
                             range_z : Tuple[float,float],
                             range_e : Tuple[float,float],
                             fit     : FitType  = FitType.unbined)->List[FitCollection2]:

    return [fit_lifetime(z, e, nbins_z, nbins_e, range_z, range_e, fit) for z,e in zip(zs,es)]

# def energy_lt_experiment_double_exp(nevt  : Number   = 1e+3,
#                                     e0    : float = 1e+4,
#                                     e1    : float = 5e+4,
#                                     lt0   : float = 2e+3,
#                                     lt1   : float = 1e+3,
#                                     std0  : float = 200,
#                                     std1  : float = 100,
#                                     zmin : float =    1,
#                                     zmax : float =  500)->Tuple[np.array, np.array]:

#     print(e0,e1,lt0,lt1,std0,std1,zmin,zmax)
#     z1, e1 = energy_lt_experiment(nevt, e0, lt0, std0, zmin, zmax)
#     z2, e2 = energy_lt_experiment(nevt, e1, lt1, std1, zmin, zmax)
#     z  = np.concatenate((z1, z2))
#     es = np.concatenate((e1, e2))
#     return z, es


def energy_lt_experiments(mexperiments : Number   = 1000,
                          nsample      : Number   = 1000,
                          e0           : float = 1e+4,
                          lt           : float = 2e+3,
                          std          : float = 0.02)->Tuple[np.array, np.array]:

    exps = [energy_lt_experiment(nsample, e0, lt, std) for i in range(int(mexperiments))]
    zs    = [x[0] for x in exps]
    es    = [x[1] for x in exps]
    return zs, es


