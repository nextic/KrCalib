import numpy as np
from pandas              import Series, DataFrame
from . core_functions    import phirad_to_deg


from typing      import List, Tuple, Dict
from . kr_types        import KrEvent
from . kr_types        import ASectorMap
from   invisible_cities.core.core_functions import in_range

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
