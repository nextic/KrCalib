import glob
import numpy  as np
import pandas as pd

from numpy.testing              import assert_allclose
from flaky                      import flaky

from . kr_parevol_functions     import cut_time_evolution
from . selection_functions      import get_time_series_df
from . kr_types                 import masks_container

from invisible_cities.io  .dst_io        import load_dst
from invisible_cities.core.testing_utils import assert_dataframes_close



def test_cut_time_evolution_different_cuts(folder_test_dst,
                                           test_dst_file ):

    dst                = load_dst(folder_test_dst + test_dst_file, "DST", "Events")
    dst                = dst.drop_duplicates('event')
    mask_S1            = np.random.choice([True, False], len(dst    ))
    mask_S2            = np.zeros_like(mask_S1, dtype=bool)
    mask_band          = np.zeros_like(mask_S1, dtype=bool)
    mask_S2[mask_S1]   = np.random.choice([True, False], sum(mask_S1))
    mask_band[mask_S2] = np.random.choice([True, False], sum(mask_S2))
    mask_cut           = masks_container(s1   = mask_S1,
                                         s2   = mask_S2,
                                         band = mask_band)
    min_time           = dst.time.min()
    max_time           = dst.time.max()
    ts, masks_time     = get_time_series_df(time_bins  = 1,
                                            time_range = (min_time, max_time),
                                            dst        = dst)
    pars               = pd.DataFrame({'ts':ts})
    pars_out           = cut_time_evolution(masks_time, dst, mask_cut, pars)
    pars_expected      = pd.DataFrame({'ts':ts, 'S1eff':0.5, 'S2eff':0.5, 'Bandeff':0.5})
    assert_dataframes_close(pars_expected, pars_out, atol=5e-2)

def test_cut_time_evolution_different_time_bins(folder_test_dst,
                                                test_dst_file ):
    dst                    = load_dst(folder_test_dst+test_dst_file, "DST", "Events")
    dst                    = dst.drop_duplicates('event')
    min_time               = dst.time.min()
    max_time               = dst.time.max()
    ts, masks_time         = get_time_series_df(time_bins  = 3,
                                                time_range = (min_time, max_time),
                                                dst        = dst)
    #set different probability of passing for different time bins, only for first mask
    probs_s1               = [0.9, 0.5, 0.1]

    mask_s1                = np.zeros(len(dst), dtype=bool)
    mask_s1[masks_time[0]] = np.random.choice([True, False], sum(masks_time[0]), p=(probs_s1[0], 1-probs_s1[0]))
    mask_s1[masks_time[1]] = np.random.choice([True, False], sum(masks_time[1]), p=(probs_s1[1], 1-probs_s1[1]))
    mask_s1[masks_time[2]] = np.random.choice([True, False], sum(masks_time[2]), p=(probs_s1[2], 1-probs_s1[2]))

    mask_cut               = masks_container(s1   = mask_s1,
                                             s2   = mask_s1,
                                             band = mask_s1)
    #the expected relative efficiencies for other cuts are 1 since the mask is repeated
    pars_expected          = pd.DataFrame({'ts':ts,
                                           'S1eff' : probs_s1,
                                           'S2eff': [1.]*3,
                                           'Bandeff':[1.]*3})
    pars                   = pd.DataFrame({'ts'   : ts})
    pars_out               = cut_time_evolution(masks_time, dst, mask_cut, pars)

    assert_dataframes_close(pars_expected, pars_out, atol=5e-2)
