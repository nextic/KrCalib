import os
import copy

import numpy as np

from pytest import fixture
from pytest import mark
from pytest import approx

from invisible_cities.reco.corrections      import read_maps
from                 .kr_types              import FitMapValue
from                 .map_functions         import add_mapinfo
from                 .map_functions         import amap_max
from                 .map_functions         import amap_min
from                 .map_functions         import amap_replace_nan_by_mean
from                 .map_functions         import amap_replace_nan_by_value


@fixture(scope='session')
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


def test_amap_max(MAPS):
    max_map = amap_max(MAPS)

    assert max_map.chi2 == approx(  814.8343078417852  )
    assert max_map.e0   == approx(12728.145825010528   )
    assert max_map.e0u  == approx(    0.733350685444711)
    assert max_map.lt   == approx( 5426.697659591667   )
    assert max_map.ltu  == approx(    9.197463879342061)


def test_amap_min(MAPS):
    min_map = amap_min(MAPS)

    assert min_map.chi2 == approx(   0.36840239825315285)
    assert min_map.e0   == approx(7115.6859368284095    )
    assert min_map.e0u  == approx(   0.2431827706912966 )
    assert min_map.lt   == approx(2583.4307547903927    )
    assert min_map.ltu  == approx(   2.674140866617924  )


@mark.parametrize("value", (-1, 0, np.pi))
def test_amap_replace_nan_by_value(MAPS, value):
    # Add a few nans
    indices = [1, 5, 7, 12, 14, 20]

    maps = copy.deepcopy(MAPS)
    maps.chi2[indices] = np.nan
    maps.e0  [indices] = np.nan
    maps.e0u [indices] = np.nan
    maps.lt  [indices] = np.nan
    maps.ltu [indices] = np.nan

    filled_nans = amap_replace_nan_by_value(maps, val = value)

    assert np.all(filled_nans.chi2[indices] == value)
    assert np.all(filled_nans.e0  [indices] == value)
    assert np.all(filled_nans.e0u [indices] == value)
    assert np.all(filled_nans.lt  [indices] == value)
    assert np.all(filled_nans.ltu [indices] == value)


@mark.parametrize("replace_nans args".split(),
                  ((amap_replace_nan_by_mean , ()  ),
                   (amap_replace_nan_by_value, (0,))))
def test_amap_replace_nan_by_something_still_contains_mapinfo(MAPS, replace_nans, args):
    maps        = add_mapinfo(MAPS, (-200, 200), (-200,200), 100, 100, 1)
    filled_nans = replace_nans(maps, *args)

    assert np.all(maps.mapinfo == filled_nans.mapinfo)
