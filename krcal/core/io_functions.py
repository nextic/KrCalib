import os
import glob
import numpy as np
import pandas as pd
import tables as tb

from   typing         import Tuple, Dict, List, Iterable
from . kr_types       import Number
from . kr_types       import ASectorMap
from . kr_types       import KrFileName
from pandas           import DataFrame, Series

from . core_functions import file_numbers_from_file_range
from pandas import Series

def filenames_from_paths(run_number  : int,
                         input_path  : str,
                         output_path : str,
                         log_path    : str,
                         trigger     : str,
                         tags        : str,
                         file_range  : Tuple[int, int])->Tuple[List[str], str, str]:
    path  = input_path
    opath = output_path
    lpath = log_path

    if file_range == "ALL":
        input_dst_filenames  = glob.glob(os.path.expandvars(f"{path}/{run_number}/kdst*.h5"))
        output_dst_filename  = os.path.expandvars(f"{opath}/dst_{run_number}_ALL.h5")
        log_filename         = os.path.expandvars(f"{lpath}/log_{run_number}_ALL.h5")
    else:
        N = file_numbers_from_file_range(file_range)

        if trigger =='':
            input_dst_filenames = [os.path.expandvars(
            f"{path}/{run_number}/kdst_{number}_{run_number}_{tags}.h5") for number in N]
        else:
            input_dst_filenames = [os.path.expandvars(
            f"{path}/{run_number}/kdst_{number}_{run_number}_{trigger}_{tags}.h5") for number in N]

        if trigger =='':
            output_dst_filename  = os.path.expandvars(
            f"{opath}/dst_{run_number}_{N[0]}_{N[-1]}.h5")

            log_filename         = os.path.expandvars(
            f"{lpath}/log_{run_number}_{N[0]}_{N[-1]}.h5")
        else:
                output_dst_filename  = os.path.expandvars(
                f"{opath}/dst_{run_number}_{trigger}_{N[0]}_{N[-1]}.h5")

                log_filename         = os.path.expandvars(
                f"{lpath}/log_{run_number}_{trigger}_{N[0]}_{N[-1]}.h5")

    return input_dst_filenames, output_dst_filename, log_filename


def file_numbers_from_file_range(file_range : Tuple[int, int])->List[str]:
    numbers = range(*file_range)
    N=[]
    for number in numbers:
        if number < 10:
            N.append(f"000{number}")
        elif 10 <= number < 100:
            N.append(f"00{number}")
        elif 100 <= number < 1000:
            N.append(f"0{number}")
        else:
            N.append(f"{number}")

    return N


def filenames_from_list(krfn : KrFileName,
                        input_path       : str,
                        output_path      : str,
                        map_path         : str)->KrFileName:

    path  = input_path
    opath = output_path
    mpath = map_path

    ifn = [os.path.expandvars(f"{path}/{file_name}") for file_name in krfn.input_file_names]
    ofn = os.path.expandvars(f"{opath}/{krfn.output_file_name}")
    mfn = os.path.expandvars(f"{mpath}/{krfn.map_file_name}")
    mts = os.path.expandvars(f"{mpath}/{krfn.map_file_name_ts}")
    efn = os.path.expandvars(f"{mpath}/{krfn.emap_file_name}")

    return KrFileName(ifn, ofn, mfn, mts, efn)



def write_monitor_vars(mdf : Series, log_filename : str):
    mdf.to_hdf(log_filename,
              key     = "LOG"  , mode         = "w",
              format  = "table", data_columns = True,
              complib = "zlib" , complevel    = 4)



def kdst_write(dst, filename):
    # Unfortunately, this method can't set a specific name to the table or its title.
    # It also includes an extra column ("index") which I can't manage to remove.
    dst.to_hdf(filename,
              key     = "DST"  , mode         = "w",
              format  = "table", data_columns = True,
              complib = "zlib" , complevel    = 4)

    # Workaround to re-establish the name of the table and its title
    with tb.open_file(filename, "r+") as f:
        f.rename_node(f.root.DST.table, "Events")
        f.root.DST.Events.title = "Events"


def write_maps(asm : ASectorMap, filename : str):

    # e0df  = pd.DataFrame.from_dict(asm.e0)
    # e0udf = pd.DataFrame.from_dict(asm.e0u)
    # ltdf  = pd.DataFrame.from_dict(asm.lt)
    # ltudf = pd.DataFrame.from_dict(asm.ltu)

    asm.e0.to_hdf(filename, key='e0', mode='w')
    asm.e0u.to_hdf(filename, key='e0u', mode='a')
    asm.lt.to_hdf(filename, key='lt', mode='a')
    asm.ltu.to_hdf(filename, key='ltu', mode='a')


def write_maps_ts(aMaps : Iterable[ASectorMap], ts: np.array, filename : str):

    assert len(ts) == len(aMaps)
    tsdf = pd.Series(ts)
    tsdf.to_hdf(filename, key='ts', mode='w')
    for i, t in enumerate(ts):
        asm = aMaps[i]

        # e0df  = pd.DataFrame.from_dict(asm.e0)
        # e0udf = pd.DataFrame.from_dict(asm.e0u)
        # ltdf  = pd.DataFrame.from_dict(asm.lt)
        # ltudf = pd.DataFrame.from_dict(asm.ltu)
        asm.e0.to_hdf(filename,  key =f'e0_{i}',  mode='a')
        asm.e0u.to_hdf(filename, key =f'e0u_{i}', mode='a')
        asm.lt.to_hdf(filename,  key =f'lt_{i}',  mode='a')
        asm.ltu.to_hdf(filename, key =f'ltu_{i}', mode='a')


def read_maps_ts(filename : str)->Tuple[Series, Dict[int, ASectorMap]]:

    tsMaps = {}
    ts = pd.read_hdf(filename, 'ts')

    for i in ts.index:
        e0  = pd.read_hdf(filename, f'e0_{i}')
        e0u = pd.read_hdf(filename, f'e0u_{i}')
        lt  = pd.read_hdf(filename, f'lt_{i}')
        ltu = pd.read_hdf(filename, f'ltu_{i}')
        tsMaps[i] = ASectorMap(None, e0, lt, e0u, ltu)
    return ts, tsMaps


def write_energy_map(em : DataFrame, filename : str):
    #e0df  = pd.DataFrame.from_dict(em)
    em.to_hdf(filename, key='e', mode='w')


def read_energy_map(filename : str)->DataFrame:
    me0  = pd.read_hdf(filename, 'e')
    return me0


def read_maps(filename : str)->Iterable[DataFrame]:
    me0  = pd.read_hdf(filename, 'e0')
    me0u = pd.read_hdf(filename, 'e0u')
    mlt  = pd.read_hdf(filename, 'lt')
    mltu = pd.read_hdf(filename, 'ltu')
    return me0, me0u, mlt, mltu
