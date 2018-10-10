import os
import pandas as pd
import numpy as np

from  invisible_cities.io.dst_io  import load_dsts
from  krcal.core.core_functions      import time_delta_from_time
from  krcal.core.analysis_functions  import kr_event
from  krcal.core.fit_lt_functions    import get_time_series
from  krcal.core.fit_lt_functions    import time_fcs
from  krcal.core.kr_types            import FitType, KrSector, MapType
from  krcal.core.analysis_functions  import event_map
from  krcal.core.analysis_functions  import select_xy_sectors
from krcal.core.fit_lt_functions    import fit_map_xy
import warnings
import pytest

@pytest.fixture(scope='session')
def DST(dst_filenames_path):


    dst           = load_dsts(dst_filenames_path, "DST", "Events")
    dst_time      = dst.sort_values('event')
    T             = dst_time.time.values
    DT            = time_delta_from_time(T)
    kge           = kr_event(dst, DT, dst.S2e, dst.S2q)
    return dst, DT, kge

@pytest.fixture(scope='session')
def time_series(DST):
    nt = 10
    dst, DT, kge = DST
    ts, masks = get_time_series(nt, DT[-1], kge)
    return nt, ts, masks


@pytest.fixture(scope='session')
def kBins():
    return np.array([-200., -120.,  -40.,   40.,  120.,  200.])



def test_get_time_series(time_series, DST):
    dst, DT, kge = DST

    nt, ts, masks = time_series
    lengths = [len(mask)for mask in masks]
    assert len(masks) == len(ts) == nt
    assert len(masks[0]) == len(kge.X) == len(dst)
    assert np.equal(lengths, len(dst) * np.ones(len(lengths))).all()


def test_time_fcs(time_series, DST):
    dst, DT, kge = DST

    nt, ts, masks = time_series

    fps = time_fcs(ts, masks, kge,
                   nbins_z = 10,
                   nbins_e = 25,
                   range_z = (50, 550),
                   range_e = (5000, 13500),
                   energy  = 'S2e',
                   fit     = FitType.profile)

    fpu = time_fcs(ts, masks, kge,
                    nbins_z = 10,
                    nbins_e = 25,
                    range_z = (50, 550),
                    range_e = (5000, 13500),
                    energy  = 'S2e',
                    fit     = FitType.unbined)
    assert np.allclose(fps.e0, fpu.e0, rtol=1e-02)
    assert np.allclose(fps.lt, fpu.lt, rtol=1e-02)


def test_select_xy_sectors(DST, kBins):
    dst, DT, kge = DST

    KRE = select_xy_sectors(dst, DT, dst.S2e.values, dst.S2q.values, kBins, kBins)
    neM = event_map(KRE)
    l = ((neM[0]/neM[4]).values > 0.8).all()
    r = ((neM[0]/neM[4]).values < 1.1).all()
    assert l & r


def test_fit_xy_map(DST, kBins):
    dst, DT, kge = DST

    def get_maps_t0(fmxy):
        pE0 = {}
        pLT = {}
        pC2 = {}
        for nx in fmxy.keys():
            pE0[nx] = [fmxy[nx][ny].e0[0] for ny in range(len(fmxy[nx]))] # notice [0] ts bin
            pLT[nx] = [fmxy[nx][ny].lt[0] for ny in range(len(fmxy[nx]))]
            pC2[nx] = [fmxy[nx][ny].c2[0] for ny in range(len(fmxy[nx]))]

            return (pd.DataFrame.from_dict(pE0),
            pd.DataFrame.from_dict(pLT),
            pd.DataFrame.from_dict(pC2))

    KRE = select_xy_sectors(dst, DT, dst.S2e.values, dst.S2q.values, kBins, kBins)
    neM = event_map(KRE)

    fpmxy = fit_map_xy(selection_map = KRE,
                       event_map     = neM,
                       n_time_bins   = 1,
                       time_diffs    = DT,
                       nbins_z       = 25,
                       nbins_e       = 50,
                       range_z       =(50, 550),
                       range_e       = (5000, 13500),
                       energy        = 'S2e',
                       fit           = FitType.profile,
                       n_min         = 100)

    mE0p, mLTp, mC2p = get_maps_t0(fpmxy)

    fumxy = fit_map_xy(selection_map = KRE,
                       event_map     = neM,
                       n_time_bins   = 1,
                       time_diffs    = DT,
                       nbins_z       = 25,
                       nbins_e       = 50,
                       range_z       =(50, 550),
                       range_e       = (5000, 13500),
                       energy        = 'S2e',
                       fit           = FitType.unbined,
                       n_min         = 100)

    mE0u, mLTu, mC2u = get_maps_t0(fumxy)
    r1 = (mLTp / mLTu).values
    l1 = np.allclose(r1, 1, rtol=1e-01)
    r2 = mE0p / mE0u
    l2 = np.allclose(r2, 1, rtol=1e-02)
    assert l1 & l2
