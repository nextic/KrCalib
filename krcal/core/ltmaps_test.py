import os
import pandas as pd
import numpy as np

from  invisible_cities.io.dst_io  import load_dsts

from  krcal.core.kr_types         import KrFileName
from  krcal.core.io_functions     import filenames_from_list
from  krcal.core.core_functions   import time_delta_from_time
from  krcal.core.analysis_functions  import kr_event
from  krcal.core.fit_lt_functions    import get_time_series
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


def test_filenames_from_list(DSTDIR, MAPSDIR, LDSTDIR,
                             dst_filenames, ldst_filename, map_filename, map_filename_ts,
                             dst_filenames_path, ldst_filename_path, map_filename_path,
                             map_filename_ts_path):

    krfn = KrFileName(dst_filenames, ldst_filename, map_filename, map_filename_ts, ' ')

    fn =filenames_from_list(krfn, DSTDIR, LDSTDIR, MAPSDIR)
    fn.input_file_names   == dst_filenames_path
    fn.output_file_name   == ldst_filename_path
    fn.map_file_name      == map_filename_path
    fn.map_file_name_ts   == map_filename_path
    fn.emap_file_name     == ' '


# def test_load_dst(DSTDIR, MAPSDIR, LDSTDIR,
#                   dst_filenames, ldst_filename, map_filename, map_filename_ts,
#                   dst_filenames_path, ldst_filename_path, map_filename_path,
#                   map_filename_ts_path):
#
#     krfn = KrFileName(dst_filenames, ldst_filename, map_filename, map_filename_ts, ' ')
#     fn =filenames_from_list(krfn, DSTDIR, LDSTDIR, MAPSDIR)
#     dst           = load_dsts(fn.input_file_names, "DST", "Events")
#     unique_events = ~dst.event.duplicated()
#     number_of_evts_full = np.count_nonzero(unique_events)
#     assert number_of_evts_full == len(dst)


def test_get_time_series(DST):
    dst, DT, kge = DST
    nt = 10

    ts, masks = get_time_series(nt, DT[-1], kge)
    lengths = [len(mask)for mask in masks]
    assert len(masks) == len(ts) == nt
    assert len(masks[0]) == len(kge.X) == len(dst)
    assert np.equal(lengths, len(dst) * np.ones(len(lengths))).all()
