import numpy as np
from invisible_cities.core .core_functions   import in_range


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

    numFailed = np.count_nonzero(~np.isnan(maps.lt))
    if numFailed > maxFailed:
        raise AbortingMapCreation('NUM FAILED FITS EXCEEDS MAX ALLOWED')
    else:
        return
