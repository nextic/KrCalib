import os
import pytest
import numpy  as np
import tables as tb

@pytest.fixture(scope = 'session')
def KRCALIB():
    return os.environ['$KRCALIB']

@pytest.fixture(scope = 'session')
def ICDATADIR():
    return '/Users/jjgomezcadenas/Projects/ICDATA'
    #return os.environ['$ICDATA']

@pytest.fixture(scope = 'session')
def DSTDIR(ICDATADIR):
    return os.path.join(ICDATADIR, "dst")

@pytest.fixture(scope = 'session')
def MAPSDIR(ICDATADIR):
    return os.path.join(ICDATADIR, "maps")

@pytest.fixture(scope = 'session')
def LDSTDIR(ICDATADIR):
    return os.path.join(ICDATADIR, "ldst")

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
def dst_filenames_path(DSTDIR):
    return [os.path.join(DSTDIR, 'dst_6284_trigger1_0000_7920.h5')]

@pytest.fixture(scope='session')
def ldst_filename_path(LDSTDIR):
    return os.path.join(LDSTDIR, 'lst_6284_trigger1_0000_7920.h5')

@pytest.fixture(scope='session')
def map_filename_path(MAPSDIR):
    return os.path.join(MAPSDIR, 'kr_maps_xy_6284.h5')


@pytest.fixture(scope='session')
def map_filename_ts_path(MAPSDIR):
    return os.path.join(MAPSDIR, 'kr_maps_xy_ts_6284.h5')
