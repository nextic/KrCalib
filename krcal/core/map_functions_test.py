import os
import pandas as pd
import numpy as np

import warnings
import pytest

from krcal.core.map_functions   import rphi_sector_map_def



def test_rphi_sector_map_def():
    rpsmf = rphi_sector_map_def(nSectors =4, rmax =200, sphi =90)
    R   = rpsmf.r
    PHI = rpsmf.phi

    for i in range(0,4):
        assert R[i]   == (0.0 + i * 50, 50.0 + i * 50)
        assert PHI[i] == [(0, 90), (90, 180), (180, 270), (270, 360)]
