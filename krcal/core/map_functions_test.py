import os
import pandas as pd
import numpy as np

import warnings
import pytest

from krcal.core.map_functions   import rphi_sector_map_def
from krcal.core.map_functions   import define_rphi_sectors



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
