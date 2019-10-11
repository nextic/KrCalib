import numpy as np
import pandas as pd

from   typing         import Iterable
from . kr_types       import ASectorMap

def write_complete_maps(asm      : ASectorMap,
                        filename : str       )->None:

    asm.chi2.to_hdf(filename, key='chi2', mode='w')
    asm.e0  .to_hdf(filename, key='e0'  , mode='a')
    asm.e0u .to_hdf(filename, key='e0u' , mode='a')
    asm.lt  .to_hdf(filename, key='lt'  , mode='a')
    asm.ltu .to_hdf(filename, key='ltu' , mode='a')
    if hasattr(asm, 'mapinfo'):
        asm.mapinfo.to_hdf(filename, key='mapinfo'       , mode='a')
    if hasattr(asm, 't_evol'):
        asm.t_evol .to_hdf(filename, key='time_evolution', mode='a')
