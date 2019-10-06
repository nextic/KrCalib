import os
import pytest


from invisible_cities.reco.corrections_new          import read_maps
from krcal.core.map_functions         import add_mapinfo

@pytest.fixture(scope='session')
def MAPS(MAPSDIR):
    map_fn = os.path.join(MAPSDIR, "kr_emap_xy_100_100_r_6346.h5")

    maps =  read_maps(filename=map_fn)
    return maps



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
