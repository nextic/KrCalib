import os
import pytest
import numpy  as np
import tables as tb

@pytest.fixture(scope = 'session')
def KRCALIB():
    krc = os.environ['KRCALIB']
    return os.path.join(krc, "test_data")


@pytest.fixture(scope = 'session')
def DSTDIR(KRCALIB):
    return os.path.join(KRCALIB, "dst")

@pytest.fixture(scope = 'session')
def MAPSDIR(KRCALIB):
    return os.path.join(KRCALIB, "maps")

@pytest.fixture(scope = 'session')
def KDSTDIR(KRCALIB):
    return os.path.join(KRCALIB, "kdst")

@pytest.fixture(scope = 'session')
def LDSTDIR(KRCALIB):
    return os.path.join(KRCALIB, "ldst")

@pytest.fixture(scope = 'session')
def config_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('configure_tests')

@pytest.fixture(scope = 'session')
def output_tmpdir(tmpdir_factory):
    return tmpdir_factory.mktemp('output_files')

@pytest.fixture(scope='session')
def dst_filenames():
    return ['dst_6284_trigger1_0000_7920.h5']

@pytest.fixture(scope='session')
def ldst_filename():
    return 'lst_6284_trigger1_0000_7920.h5'

@pytest.fixture(scope='session')
def map_filename():
    return  'kr_maps_xy_6284.h5'

@pytest.fixture(scope='session')
def map_filename_ts():
    return 'kr_maps_xy_ts_6284.h5'

@pytest.fixture(scope='session')
def dst_filenames_path(DSTDIR, dst_filenames):
    return [os.path.join(DSTDIR, dst_filenames[0])]

@pytest.fixture(scope='session')
def ldst_filename_path(LDSTDIR, ldst_filename):
    return os.path.join(LDSTDIR, ldst_filename)

@pytest.fixture(scope='session')
def map_filename_path(MAPSDIR, map_filename):
    return os.path.join(MAPSDIR, map_filename)

@pytest.fixture(scope='session')
def map_filename_ts_path(MAPSDIR, map_filename_ts):
    return os.path.join(MAPSDIR, map_filename_ts)
