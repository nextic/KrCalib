import os
import pytest
import numpy  as np
import tables as tb
import pandas as pd

@pytest.fixture(scope = 'session')
def ICARO():
    krc = os.environ['ICARO']
    return os.path.join(krc, "test_data")


@pytest.fixture(scope = 'session')
def DSTDIR(ICARO):
    return os.path.join(os.environ['IC_DATA'], "dst")

@pytest.fixture(scope = 'session')
def MAPSDIR(ICARO):
    return os.path.join(ICARO, "maps")

@pytest.fixture(scope = 'session')
def KDSTDIR(ICARO):
    return os.path.join(ICARO, "kdst")

@pytest.fixture(scope = 'session')
def LDSTDIR(ICARO):
    return os.path.join(ICARO, "ldst")

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

@pytest.fixture(scope='session')
def dstData():
    # define dst
    D = {}
    D['X']      = np.random.random(20) * 100
    D['Y']      = np.random.random(20) * 100
    D['Z']      = np.random.random(20) * 100
    D['R']      = np.random.random(20) * 100
    D['Phi']    = np.random.random(20) * 100
    D['S2e']    = np.random.random(20) * 100
    D['S1e']    = np.random.random(20) * 100
    D['S2q']    = np.random.random(20) * 100
    D['time']   = np.arange(0,100,5)
    dst         = pd.DataFrame.from_dict(D)

    ### Define x & y bins
    xb = np.arange(0,101,25)
    yb = np.arange(0,101,25)
    nbx = len(xb) -1
    nby = len(yb) -1

    # define time bins
    nt = 5
    t0 = dst.time.values[0]
    tf = dst.time.values[-1]

    step = int((tf -  t0) / nt)
    indx = [(0, 19), (19, 38), (38, 57), (57, 76), (76, 95)]
    ts   = [9.5, 28.5, 47.5, 66.5, 85.5]
    return dst, xb, yb, nbx, nby, nt, t0, tf, step, indx, ts
