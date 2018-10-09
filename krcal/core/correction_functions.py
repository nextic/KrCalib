import numpy as np
from pandas              import Series, DataFrame
from . core_functions    import phirad_to_deg


from typing      import List, Tuple, Dict
from . kr_types        import KrEvent
from . kr_types        import ASectorMap
from   invisible_cities.core.core_functions import in_range

import sys
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
    Returns the energy corrected by geometry:
        E is the raw energy (S2e, or Q)
        X, Y cartesian coordinates
        EM is the E0 map
        xr, yr are the ranges of X,Y (e.g, (-220,220))
        nx, ny the number of bins in (x, y)

    """

    CE = xy_correction_matrix(X, Y, E0M, xr, yr, nx, ny)
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
    """ Returns the energy corrected by LT """

    LT = xy_correction_matrix(X, Y, LTM, xr, yr, nx, ny)
    return E * np.exp(Z / LT)


def e0_xy_correction_ts(kh     : KrEvent,
                        tts    : Series,
                        tsMaps : Dict[int, ASectorMap],
                        xr     : Tuple[int, int],
                        yr     : Tuple[int, int],
                        nx     : int,
                        ny     : int)->KrEvent:

    ts = tts.values
    masks = get_time_selection_masks(kh.DT, tts.values)
    kcts = get_krevent_time_series(kh, masks)
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



def lt_xy_correction_ts(kh : KrEvent,
                        tts    : Series,
                        tsMaps : Dict[int, ASectorMap],
                        xr     : Tuple[int, int],
                        yr     : Tuple[int, int],
                        nx     : int,
                        ny     : int)->KrEvent:


    ts = tts.values
    masks = get_time_selection_masks(kh.DT, tts.values)
    kcts = get_krevent_time_series(kh, masks)
    LTMs =[tsMaps[i].lt for i in tts.index]

    EE = []
    for j, kct in enumerate(kcts):
        logging.info(f' lt_xy_correction_ts: time sector = {j}')

        LTM = LTMs[j]
        ec  = lt_xy_correction(kct.E, kct.X, kct.Y, kct.Z, LTM, xr, yr, nx, ny)
        logging.info(f' Correction vector (average):  = {np.mean(ec)}')
        EE.append(ec)

    E = np.concatenate(EE)

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



def e0_correction(E   : np.array,
                  R   : np.array,
                  PHI : np.array,
                  CE  : DataFrame,
                  fr   : float,
                  fphi : float)->np.array:
    """
    Returns the energy corrected by geometry:
        E is the raw energy (S2e, or Q)
        R is the Radial coordinate
        Phi is the angular Phi coordinate (values between -pi and pi)
        CE is the correction matrix
        fr is the radial factor corresponding to Rmax / ns, where:
           Rmax is maximum radius in the CE (eg, 180 mm)
           ns is the number of sectors (e.g, 10, then fr = 18)
        fphi is the phi factor corresponding to 360 / nw where
           nw is the number of phi sectors (wedges). For wedges each 15 deg
           then nw = 24 and fphi = 15

    """

    I = get_rphi_indexes(R, PHI, fr, fphi)
    ce = np.array([CE[i[0]][i[1]] for i in I])
    return E / ce


def lt_correction(E    : np.array,
                  R    : np.array,
                  PHI  : np.array,
                  Z    : np.array,
                  CLT  : DataFrame,
                  fr   : float,
                  fphi : float)->np.array:
    """
    Returns the energy corrected by LT:
            E is the energy corrected by geometry
            R, Phi radial and azhimutal coordinates i
            CLT is the correction matrix

        """

    I = get_rphi_indexes(R, PHI, fr, fphi)
    LT = np.array([CLT[i[0]][i[1]] for i in I])
    return E * np.exp(Z / LT)




def lt_correction_ts(kh     : KrEvent,
                     tts    : Series,
                     tsMaps : Dict[int, ASectorMap],
                     fr     : float,
                     fphi   : float)->KrEvent:
    """
    Returns the energy corrected by LT:
            E is the energy corrected by geometry
            R, Phi radial and azhimutal coordinates i
            CLT is the correction matrix

        """
    ts = tts.values
    masks = [in_range(kh.DT, ts[i], ts[i+1]) for i in range(len(ts)-1)]
    masks.append(in_range(kh.DT, ts[-1], kh.DT[-1]))

    kcts = [KrEvent(X   = kh.X[sel_mask],
                    Y   = kh.Y[sel_mask],
                    Z   = kh.Z[sel_mask],
                    R   = kh.R[sel_mask],
                    Phi = kh.Phi[sel_mask],
                    T   = kh.T[sel_mask],
                    DT  = kh.DT[sel_mask],
                    S2e = kh.S2e[sel_mask],
                    S1e = kh.S1e[sel_mask],
                    S2q = kh.S2q[sel_mask],
                    E  = kh.E[sel_mask],
                    Q  = kh.Q[sel_mask]) for sel_mask in masks]

    CLTs =[tsMaps[i].lt for i in tts.index]

    #print(f'len(kcts) ={len(kcts)}, len(CLTs) = {len(CLTs)}')
    EE = []
    for j, kct in enumerate(kcts):
        print(kct.E.shape)

        CLT = CLTs[j]
        I = get_rphi_indexes(kct.R, kct.Phi, fr, fphi)
        #print(j)
        #print(CLT)


        LT = np.array([CLT[i[0]][i[1]] for i in I])

        #print(kct.E[0:10])
        #print(LT[0:10])
        Ec = kct.E * np.exp(kct.Z / LT)
        #print(Ec[0:10])
        EE.append(Ec)


    X = np.concatenate([kct.X for kct in kcts])
    Y = np.concatenate([kct.Y for kct in kcts])
    Z = np.concatenate([kct.Z for kct in kcts])
    R = np.concatenate([kct.R for kct in kcts])
    Phi = np.concatenate([kct.Phi for kct in kcts])
    T = np.concatenate([kct.T for kct in kcts])
    DT = np.concatenate([kct.DT for kct in kcts])
    S2e = np.concatenate([kct.S2e for kct in kcts])
    S1e = np.concatenate([kct.S1e for kct in kcts])
    S2q = np.concatenate([kct.S2q for kct in kcts])
    E = np.concatenate(EE)
    Q = np.concatenate([kct.Q for kct in kcts])

    return KrEvent(X   = X,
                    Y   = Y,
                    Z   = Z,
                    R   = R,
                    Phi = Phi,
                    T   = T,
                    DT  = DT,
                    S2e = S2e,
                    S1e = S1e,
                    S2q = S2q,
                    E  = E,
                    Q  = Q)



def e0_correction_ts(kh     : KrEvent,
                     tts    : Series,
                     tsMaps : Dict[int, ASectorMap],
                     fr     : float,
                     fphi   : float)->KrEvent:
    """
    Returns the energy corrected by LT:
            E is the energy corrected by geometry
            R, Phi radial and azhimutal coordinates i
            CLT is the correction matrix

        """

    ts = tts.values
    masks = [in_range(kh.DT, ts[i], ts[i+1]) for i in range(len(ts)-1)]
    masks.append(in_range(kh.DT, ts[-1], kh.DT[-1]))

    kcts = [KrEvent(X   = kh.X[sel_mask],
                    Y   = kh.Y[sel_mask],
                    Z   = kh.Z[sel_mask],
                    R   = kh.R[sel_mask],
                    Phi = kh.Phi[sel_mask],
                    T   = kh.T[sel_mask],
                    DT  = kh.DT[sel_mask],
                    S2e = kh.S2e[sel_mask],
                    S1e = kh.S1e[sel_mask],
                    S2q = kh.S2q[sel_mask],
                    E  = kh.E[sel_mask],
                    Q  = kh.Q[sel_mask]) for sel_mask in masks]

    CLTs =[tsMaps[i].e0 for i in tts.index]

    #print(f'len(kcts) ={len(kcts)}, len(CLTs) = {len(CLTs)}')
    EE = []
    for j, kct in enumerate(kcts):
        print(f'time sector {j}')

        CLT = CLTs[j]
        mu = (CLT.mean()).mean()
        CE = CLT / mu
        I = get_rphi_indexes(kct.R, kct.Phi, fr, fphi)
        #print(j)
        #print(CLT)


        ce = np.array([CE[i[0]][i[1]] for i in I])

        #print(kct.E[0:10])
        #print(LT[0:10])
        Ec = kct.E / ce
        #print(Ec[0:10])
        EE.append(Ec)

    E = np.concatenate(EE)

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


def get_rphi_indexes(R    : np.array,
                     PHI  : np.array,
                     fr   : float,
                     fphi : float)->List[Tuple[int,int]]:

    r_i = (R / fr).astype(int)
    phi_i = (phirad_to_deg(PHI) / fphi).astype(int)
    return list(zip(r_i, phi_i))



def xy_correction_matrix(X  : np.array,
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
    vx = sizeof_xy_voxel(xr, nx)
    vy = sizeof_xy_voxel(yr, ny)

    I = get_xy_indexes(X, Y, abs(xr[0]), abs(yr[0]), vx, vy)
    return np.array([C[i[0]][i[1]] for i in I])


def get_xy_indexes(X  : np.array,
                   Y  : np.array,
                   x0 : float,
                   y0 : float,
                   fx : float,
                   fy : float)->List[Tuple[int,int]]:


    x_i = ((X + x0) / fx).astype(int)
    y_i = ((Y + y0) / fy).astype(int)
    return list(zip(x_i, y_i))


def sizeof_xy_voxel(rxy, nxy):
    """rxy = (x0, x1) defines de interval in x (y), e.g, (x0, x1) = (-220, 220)
        nxy is the number of bins in x (y).
        then, the interval is shifted to positive values and divided by number of bins:
        x0' --> abs(x0) ; x1' = x0' + x1
        fr = x1' / n

    """
    x0 = abs(rxy[0])
    x1 = rxy[1] + x0
    fr = x1 / nxy
    return fr


def get_time_selection_masks(DT : np.array, ts : np.array)->List[np.array]:
    masks = [in_range(DT, ts[i], ts[i+1]) for i in range(len(ts)-1)]
    masks.append(in_range(DT, ts[-1], DT[-1]))
    return masks


def get_krevent_time_series(kh : KrEvent, masks : List[np.array])->List[KrEvent]:
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
                    E  = kh.E[sel_mask],
                    Q  = kh.Q[sel_mask]) for sel_mask in masks]
