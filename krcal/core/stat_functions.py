import numpy as np
from   numpy                                   import sqrt
from   invisible_cities.core.core_functions    import in_range
from   typing                                  import Tuple
from . core_functions                          import  NN
from . kr_types                                import Number


def relative_error_ratio(a : float, sigma_a: float, b :float, sigma_b : float) ->float:
    return sqrt((sigma_a / a)**2 + (sigma_b / b)**2)


def mean_and_std(x : np.array, range_ : Tuple[Number, Number])->Tuple[Number, Number]:
    """Computes mean and std for an array within a range: takes into account nans"""

    mu = NN
    std = NN

    if all(np.isnan(x)):  # all elements are nan
        mu  = NN
        std  = NN
    else:
        x_nonnan = x[np.isfinite(x)]
        y = x_nonnan[in_range(x_nonnan, *range_)]
        if len(y) == 0:
            print(f'warning, empty slice of x = {x} in range = {range_}')
            mu = NN
            std = NN
        else:
            mu = np.mean(y)
            std = np.std(y)

    return mu, std




