import numpy as np
from invisible_cities.core .core_functions import in_range
from invisible_cities.reco.corrections_new import ASectorMap



class AbortingMapCreation(Exception):
    pass


def check_if_values_in_interval(values          : np.array,
                                low_lim         : float   ,
                                up_lim          : float   ,
                                raising_message : str = ''
                                )->None:
    """
    Raises exception, aborting kr map computation, if input
    values are not all inside the interval (low_lim, up_lim).
    Parameters
    ----------
    values : np.array
        Input array to check.
    low_lim: float
        Lower limit of the interval.
    up_lim: float
        Upper limit of the interval.
    raising_message: string
        Message to print if exception raises.
    Returns
    ----------
        None if values are in the interval. Otherwise, it raises an exception.
    """
    if in_range(values, low_lim, up_lim).all():
        return;
    else:
        raise AbortingMapCreation(raising_message)

def check_failed_fits(maps : ASectorMap, maxFailed : float = 600 ):
def get_core(nbins,rmax, rfid):
    r = np.zeros([nbins,nbins])
    x = np.linspace(-rmax, rmax, nbins)
    for i in range(nbins):
        for j in range(nbins):
            r[i,j] = (x[i]**2 + x[j]**2)**0.5
    mask = in_range(r, 0, rfid)
    return mask

    numFailed = np.count_nonzero(~np.isnan(maps.lt))
    if numFailed > maxFailed:
        message = "Number of failed fits ({0}) ".format(numFailed)
        message += "exceeds max. allowed ({0}).".format(maxFailed)
        raise AbortingMapCreation(message)
    else:
        return
