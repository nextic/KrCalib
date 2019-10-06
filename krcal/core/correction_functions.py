"""Module correction_functions.
This module includes the functions needed to correct energy (lifetime and geom corrections).

Notes
-----
    KrCalib code depends on the IC library.
    Public functions are documented using numpy style convention

Documentation
-------------
    Insert documentation https
"""


import numpy as np
from   invisible_cities.core.core_functions import in_range

from   pandas            import DataFrame
from . core_functions    import phirad_to_deg
from typing              import List, Tuple

import logging
log = logging.getLogger()

def e0_xy_correction(E   : np.array,
                     X   : np.array,
                     Y   : np.array,
                     E0M : DataFrame,
                     xr  : Tuple[int, int],
                     yr  : Tuple[int, int],
                     nx  : int,
                     ny  : int)->np.array:
    """
    Computes the energy vector corrected by geometry in bins of XY.

    Parameters
    ----------
    E
        The uncorrected energy vector.
    X
        Array of X bins.
    Y
        Array of Y bins.
    E0M
        Map of geometrical corrections (E0 map).
    xr
        Range of X (e.g, (-220,220)).
    yr
        Range of Y (e.g, (-220,220)).
    nx
        Number of bins in X.
    ny
        Number of bins in Y.

    Returns
    -------
    np.array
        The corrected energy vector (by energy).

    """
    CE = xy_correction_matrix_(X, Y, E0M, xr, yr, nx, ny)
    return E / CE


def lt_xy_correction(E    : np.array,
                     X    : np.array,
                     Y    : np.array,
                     Z    : np.array,
                     LTM  : DataFrame,
                     xr   : Tuple[int, int],
                     yr   : Tuple[int, int],
                     nx   : int,
                     ny   : int)->np.array:
    """
    Computes the energy vector corrected by lifetime in bins of XY.

    Parameters
    ----------
    E
        The uncorrected energy vector.
    X
        Array of X bins.
    Y
        Array of Y bins.
    LTM
        Map of lifetime corrections (LT map).
    xr
        Range of X (e.g, (-220,220)).
    yr
        Range of Y (e.g, (-220,220)).
    nx
        Number of bins in X.
    ny
        Number of bins in Y.

    Returns
    -------
    np.array
        The corrected energy vector (by lifetime).

    """
    LT = xy_correction_matrix_(X, Y, LTM, xr, yr, nx, ny)
    return E * np.exp(Z / LT)


def e0_rphi_correction(E    : np.array,
                       R    : np.array,
                       PHI  : np.array,
                       CE   : DataFrame,
                       fr   : float,
                       fphi : float)->np.array:
    """
    Computes the energy vector corrected by geometry in bins of RPhi.

    Parameters
    ----------
    E
        The uncorrected energy vector.
    R
        Array of R bins.
    PHI
        Array of PHI bins.
    CE
        Map of geometrical corrections (E0 map).
    fr
        Radial factor: fr = RMAX / nsectors, where RMAX is the maximum radius
        used to compute the RPHI map and nsectors is the number of radial sectors.
        For example, for RMAX = 200 and 10 radial sectors, fr = 20.
    fphi
        Phi factor, equal to the size of the phi wedges used in the map in degrees.
        For example, if the map has computed 10 phi wedges, each wedge has 36 degrees
        (360 / 10) and thus fphi = 36

    Returns
    -------
    np.array
        The corrected energy vector (by energy).

    """
    I = get_rphi_indexes_(R, PHI, fr, fphi)
    ce = np.array([CE[i[0]][i[1]] for i in I])
    return E / ce


def lt_rphi_correction(E    : np.array,
                       R    : np.array,
                       PHI  : np.array,
                       Z    : np.array,
                       CLT  : DataFrame,
                       fr   : float,
                       fphi : float)->np.array:
    """
    Computes the energy vector corrected by geometry in bins of RPhi.

    Parameters
    ----------
    E
        The uncorrected energy vector.
    R
        Array of R bins.
    PHI
        Array of PHI bins.
    CLT
        Map of geometrical corrections (LT map).
    fr
        Radial factor: fr = RMAX / nsectors, where RMAX is the maximum radius
        used to compute the RPHI map and nsectors is the number of radial sectors.
        For example, for RMAX = 200 and 10 radial sectors, fr = 20.
    fphi
        Phi factor, equal to the size of the phi wedges used in the map in degrees.
        For example, if the map has computed 10 phi wedges, each wedge has 36 degrees
        (360 / 10) and thus fphi = 36

    Returns
    -------
    np.array
        The corrected energy vector (by lifetime).

    """
    I = get_rphi_indexes_(R, PHI, fr, fphi)
    LT = np.array([CLT[i[0]][i[1]] for i in I])
    return E * np.exp(Z / LT)






def get_rphi_indexes_(R    : np.array,
                      PHI  : np.array,
                      fr   : float,
                      fphi : float)->List[Tuple[int,int]]:
    """Returns a list of pairs of ints (r_i, phi_i)"""
    r_i = (R / fr).astype(int)
    phi_i = (phirad_to_deg(PHI) / fphi).astype(int)
    return list(zip(r_i, phi_i))


def xy_correction_matrix_(X  : np.array,
                         Y  : np.array,
                         C  : DataFrame,
                         xr : Tuple[int, int],
                         yr : Tuple[int, int],
                         nx : int,
                         ny : int)->np.array:
    """
    Returns a correction matrix in XY computed from the
    map represented by the DataFrame C:

    """
    vx = sizeof_xy_voxel_(xr, nx)
    vy = sizeof_xy_voxel_(yr, ny)
    I = get_xy_indexes_(X, Y, abs(xr[0]), abs(yr[0]), vx, vy)
    return np.array([C[i[0]][i[1]] for i in I])


def get_xy_indexes_(X  : np.array,
                    Y  : np.array,
                    x0 : float,
                    y0 : float,
                    fx : float,
                    fy : float)->List[Tuple[int,int]]:
    """Returns a list of pairs of ints, (x_i, y_i)"""
    x_i = ((X + x0) / fx).astype(int)
    y_i = ((Y + y0) / fy).astype(int)
    return list(zip(x_i, y_i))


def sizeof_xy_voxel_(rxy : Tuple[int,int], nxy : int)->float:
    """
    rxy = (x0, x1) defines de interval in x (y), e.g, (x0, x1) = (-220, 220)
    nxy is the number of bins in x (y).
    then, the interval is shifted to positive values and divided by number of bins:
    x0' --> abs(x0) ; x1' = x0' + x1
    fr = x1' / n

    """
    x0 = abs(rxy[0])
    x1 = rxy[1] + x0
    fr = x1 / nxy
    return fr


def get_time_selection_masks_(DT : np.array, ts : np.array)->List[np.array]:
    """Returns a list of selection masks"""
    masks = [in_range(DT, ts[i], ts[i+1]) for i in range(len(ts)-1)]
    masks.append(in_range(DT, ts[-1], DT[-1]))
    return masks
