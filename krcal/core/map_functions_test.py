import os
import pandas as pd
import numpy as np

import warnings
import pytest

from krcal.core.rphi_maps_functions   import rphi_sector_map_def
from krcal.core.rphi_maps_functions   import define_rphi_sectors
from krcal.core.io_functions          import read_maps
from krcal.core.map_functions         import add_mapinfo

@pytest.fixture(scope='session')
def MAPS(MAPSDIR):
    map_fn = os.path.join(MAPSDIR, "kr_emap_xy_100_100_r_6346.h5")

    maps =  read_maps(filename=map_fn)
    return maps


def test_rphi_sector_map_def():
    rpsmf = rphi_sector_map_def(nSectors =4, rmax =200, sphi =90)
    R   = rpsmf.r
    PHI = rpsmf.phi

    for i in range(0,4):
        assert R[i]   == (0.0 + i * 50, 50.0 + i * 50)
        assert PHI[i] == [(0, 90), (90, 180), (180, 270), (270, 360)]


def test_define_rphi_sectors():
    rpsmf = rphi_sector_map_def(nSectors =4, rmax =200, sphi =90)
    rps = define_rphi_sectors(rpsmf)

    for i in range(0,4):
        krsl = rps[i]
        for j, krs in enumerate(krsl):
            assert krs.rmin == i * 50
            assert krs.rmax == 50 + i * 50
            assert krs.phimin == j * 90
            assert krs.phimax == 90 + j * 90

def test_define_mapinfo(MAPS):
    asm = MAPS
    atest = add_mapinfo(asm, (-200, 200), (-200,200), 100, 100, 1)
    assert atest.mapinfo.xmin       == -200
    assert atest.mapinfo.xmax       ==  200
    assert atest.mapinfo.ymin       == -200
    assert atest.mapinfo.ymax       ==  200
    assert atest.mapinfo.nx         ==  100
    assert atest.mapinfo.ny         ==  100
    assert atest.mapinfo.run_number ==    1
