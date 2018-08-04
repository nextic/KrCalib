import numpy as np
import matplotlib.pyplot as plt
from invisible_cities.core.core_functions import in_range
from typing                               import Tuple, List
from . kr_types                           import Number

def mean_and_std(x : np.array, range_ : Tuple[Number, Number])->Tuple[Number, Number]:
    """Computes mean and std for an array within a range"""

    mu = np.mean(x[in_range(x, *range_)])
    std = np.std(x[in_range(x, *range_)])
    return mu, std


def gaussian_experiment(nevt : int = 1e+3, mean: float = 100, std: float = 10)->np.array:
    Nevt  = int(nevt)
    e  = np.random.normal(mean, std, Nevt)
    return e


def run_gaussian_experiments(mexperiments : int   = 1000,
                             nsample      : int   = 1000,
                             mean         : float = 1e+4,
                             std          : float = 100)->List[np.array]:
    return [gaussian_experiment(nsample, mean, std) for i in range(mexperiments)]



def gaussian_experiments_with_variable_mean_and_std(mexperiments : int   = 1000,
                                                    nsample      : int   = 1000,
                             mean         : float = 1e+4,
                             std          : float = 100)->List[np.array]:
Nevt  = int(1e3)
sigmas = np.random.uniform(low=1.0, high=50., size=100)
means  = np.random.uniform(low=100, high=1000., size=100)

SEED = []
MU = []
STD = []
AVG = []
RMS = []
CHI2 = []
n_sigma =3
for sigma in sigmas:
    for mean in means:
        SEED.append(Measurement(mean, sigma))
        e  = np.random.normal(mean, sigma, Nevt)
        r = mean - n_sigma * sigma, mean + n_sigma * sigma
        bin_size = (r[1] - r[0]) / 50
        gp = gaussian_parameters(e, range = r, bin_size=bin_size)
        try:
            fc = fit_energy(e, nbins=50, range=r, n_sigma = n_sigma)
        except TypeError:
            print(f'fit failed with TypeError for mean = {mean}, sigma ={sigma}')
            raise

        MU.append(Measurement(fc.fr.par[1], fc.fr.err[1]))
        STD.append(Measurement(fc.fr.par[2], fc.fr.err[2] ))
        AVG.append(gp.mu)
        RMS.append(gp.std)

        CHI2.append(fc.fr.chi2)

print(len(MU))


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


def run_energy_lt_experiments(mexperiments : Number   = 1000,
                             nsample       : Number   = 1000,
                             e0            : float = 1e+4,
                             lt            : float = 2e+3,
                             std           : float = 0.02)->Tuple[np.array, np.array]:

    exps = [energy_lt_experiment(nsample, e0, lt, std) for i in range(int(mexperiments))]
    zs    = [x[0] for x in exps]
    es    = [x[1] for x in exps]
    return zs, es
