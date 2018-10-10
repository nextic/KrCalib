"""
Tests for fit_functions
"""

import numpy as np

from pytest        import mark
from pytest        import approx
from pytest        import raises
from flaky         import flaky
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from hypothesis            import given, settings
from hypothesis.strategies import integers
from hypothesis.strategies import floats
from invisible_cities.core.testing_utils       import exactly
from invisible_cities.core.testing_utils       import float_arrays
from invisible_cities.core.testing_utils       import FLOAT_ARRAY
from invisible_cities.core.testing_utils       import random_length_float_arrays

from . io_functions                  import filenames_from_paths
from  krcal.core.io_functions        import filenames_from_list
from  krcal.core.kr_types            import KrFileName


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



def test_filenames_from_paths():
    input_dst_filenames, output_dst_filename, log_filename = filenames_from_paths(run_number=6261,
                                                                              input_path ="/input",
                                                                              output_path='/out',
                                                                              log_path   ='/log',
                                                                              trigger ='trigger1',
                                                                              tags    ='v1',
                                                                              file_range=(0,10))

    assert input_dst_filenames[0]  == '/input/6261/kdst_0000_6261_trigger1_v1.h5'
    assert input_dst_filenames[-1] == '/input/6261/kdst_0009_6261_trigger1_v1.h5'
    assert output_dst_filename     == '/out/dst_6261_trigger1_0000_0009.h5'
    assert log_filename            == '/log/log_6261_trigger1_0000_0009.h5'

def test_filenames_from_paths_trigger_empty():
    input_dst_filenames, output_dst_filename, log_filename = filenames_from_paths(run_number=6261,
                                                                              input_path ="/input",
                                                                              output_path='/out',
                                                                              log_path   ='/log',
                                                                              trigger ='',
                                                                              tags    ='v1',
                                                                              file_range=(0,10))

    assert input_dst_filenames[0]  == '/input/6261/kdst_0000_6261_v1.h5'
    assert input_dst_filenames[-1] == '/input/6261/kdst_0009_6261_v1.h5'
    assert output_dst_filename     == '/out/dst_6261_0000_0009.h5'
    assert log_filename            == '/log/log_6261_0000_0009.h5'
