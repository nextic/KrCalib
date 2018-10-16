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

import sys
import numpy as np
from   invisible_cities.core.core_functions import in_range

from   pandas            import Series, DataFrame
from . core_functions    import phirad_to_deg
from typing              import List, Tuple, Dict
from . kr_types          import KrEvent
from . kr_types          import ASectorMap

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


def e0_xy_correction_ts(kh     : KrEvent,
                        tts    : Series,
                        tsMaps : Dict[int, ASectorMap],
                        xr     : Tuple[int, int],
                        yr     : Tuple[int, int],
                        nx     : int,
                        ny     : int)->KrEvent:
    """
    Computes the energy vector corrected by geometry in bins of XY for the time series tts.

    Parameters
    ----------
    kh
        Input KrEvent.
    tts
        Time series.
    tsMaps
        A dictionary of E0M. Each element of tsMaps corresponds to a time in tts.
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
    KrEvent
        A KrEvent including the corrected energy vector (by energy).

    """
    masks = get_time_selection_masks_(kh.DT, tts.values)
    kcts = get_krevent_time_series_(kh, masks)
    E0Ms =[tsMaps[i].e0 for i in tts.index]

    EE = []
    for j, kct in enumerate(kcts):
        logging.info(f' e0_xy_correction_ts: time sector = {j}')
        E0m = E0Ms[j]
        mx = (E0m.max()).max()
        E0M = E0m / mx
        ec = e0_xy_correction(kct.E, kct.X, kct.Y, E0M, xr, yr, nx, ny)
        logging.info(f' Correction vector (average):  = {np.mean(ec)}')
        EE.append(ec)
    E = np.concatenate(EE)
    return concatenate_kr_event_(kcts, E)


def lt_xy_correction_ts(kh     : KrEvent,
                        tts    : Series,
                        tsMaps : Dict[int, ASectorMap],
                        xr     : Tuple[int, int],
                        yr     : Tuple[int, int],
                        nx     : int,
                        ny     : int)->KrEvent:
    """
    Computes the energy vector corrected by lifetime in bins of XY for the time series tts.

    Parameters
    ----------
    kh
        Input KrEvent.
    tts
        Time series.
    tsMaps
        A dictionary of E0M. Each element of tsMaps corresponds to a time in tts.
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
    KrEvent
        A KrEvent including the corrected energy vector (by lifetime).

    """
    masks = get_time_selection_masks_(kh.DT, tts.values)
    kcts = get_krevent_time_series_(kh, masks)
    LTMs =[tsMaps[i].lt for i in tts.index]

    EE = []
    for j, kct in enumerate(kcts):
        logging.info(f' lt_xy_correction_ts: time sector = {j}')
        LTM = LTMs[j]
        ec  = lt_xy_correction(kct.E, kct.X, kct.Y, kct.Z, LTM, xr, yr, nx, ny)
        logging.info(f' Correction vector (average):  = {np.mean(ec)}')
        EE.append(ec)
    E = np.concatenate(EE)
    return concatenate_kr_event_(kcts, E)


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


def lt_rphi_correction_ts(kh     : KrEvent,
                          tts    : Series,
                          tsMaps : Dict[int, ASectorMap],
                          fr     : float,
                          fphi   : float)->KrEvent:
    """
    Computes the energy vector corrected by lifetime in bins of RPHI for the time series tts.

    Parameters
    ----------
    kh
        Input KrEvent.
    tts
        Time series.
    tsMaps
        A dictionary of E0M. Each element of tsMaps corresponds to a time in tts.
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
    KrEvent
        A KrEvent including the corrected energy vector (by lifetime).

    """
    masks = get_time_selection_masks_(kh.DT, tts.values)
    kcts = get_krevent_time_series_(kh, masks)
    CLTs =[tsMaps[i].lt for i in tts.index]

    EE = []
    for j, kct in enumerate(kcts):
        CLT = CLTs[j]
        I = get_rphi_indexes_(kct.R, kct.Phi, fr, fphi)
        LT = np.array([CLT[i[0]][i[1]] for i in I])
        Ec = kct.E * np.exp(kct.Z / LT)
        EE.append(Ec)
    E = np.concatenate(EE)
    return concatenate_kr_event_(kcts, E)


def e0_rphi_correction_ts(kh     : KrEvent,
                          tts    : Series,
                          tsMaps : Dict[int, ASectorMap],
                          fr     : float,
                          fphi   : float)->KrEvent:
    """
    Computes the energy vector corrected by geometry in bins of RPHI for the time series tts.

    Parameters
    ----------
    kh
        Input KrEvent.
    tts
        Time series.
    tsMaps
        A dictionary of E0M. Each element of tsMaps corresponds to a time in tts.
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
    KrEvent
        A KrEvent including the corrected energy vector (by geometry).

    """
    masks = get_time_selection_masks_(kh.DT, tts.values)
    kcts = get_krevent_time_series_(kh, masks)
    CLTs =[tsMaps[i].e0 for i in tts.index]

    EE = []
    for j, kct in enumerate(kcts):
        logging.debug(f'time sector {j}')

        CLT = CLTs[j]
        mu = (CLT.max()).max()
        CE = CLT / mu
        I = get_rphi_indexes_(kct.R, kct.Phi, fr, fphi)
        ce = np.array([CE[i[0]][i[1]] for i in I])
        Ec = kct.E / ce
        EE.append(Ec)

    E = np.concatenate(EE)
    return concatenate_kr_event_(kcts, E)


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


def get_krevent_time_series_(kh : KrEvent, masks : List[np.array])->List[KrEvent]:
    """Returns a list of KrEvents computed from a KrEvent and a list of masks"""
    return [KrEvent(X   = kh.X[sel_mask],
                    Y   = kh.Y[sel_mask],
                    Z   = kh.Z[sel_mask],
                    R   = kh.R[sel_mask],
                    Phi = kh.Phi[sel_mask],
                    T   = kh.T[sel_mask],
                    DT  = kh.DT[sel_mask],
                    S2e = kh.S2e[sel_mask],
                    S1e = kh.S1e[sel_mask],
                    S2q = kh.S2q[sel_mask],
                    E   = kh.E[sel_mask],
                    Q   = kh.Q[sel_mask]) for sel_mask in masks]


def concatenate_kr_event_(kcts : List[KrEvent], E : np.array)->KrEvent:
    """Concatenates a list of KrEvent into a new KrEvent"""
    return KrEvent(X   = np.concatenate([kct.X for kct in kcts]),
                   Y   = np.concatenate([kct.Y for kct in kcts]),
                   Z   = np.concatenate([kct.Z for kct in kcts]),
                   R   = np.concatenate([kct.R for kct in kcts]),
                   Phi = np.concatenate([kct.Phi for kct in kcts]),
                   T   = np.concatenate([kct.T for kct in kcts]),
                   DT  = np.concatenate([kct.DT for kct in kcts]),
                   S2e = np.concatenate([kct.S2e for kct in kcts]),
                   S1e = np.concatenate([kct.S1e for kct in kcts]),
                   S2q = np.concatenate([kct.S2q for kct in kcts]),
                   E  = E,
                   Q  = np.concatenate([kct.Q for kct in kcts]))
