import os
import glob
import numpy as np

from   typing         import Tuple, List, Iterable
from . kr_types       import Number
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
            f"{path}/{run_number}/kdst_{number}_{run_number}_{tags}_krth.h5") for number in N]
        else:
            input_dst_filenames = [os.path.expandvars(
            f"{path}/{run_number}/kdst_{number}_{run_number}_{trigger}_{tags}_krth.h5") for number in N]

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


def filenames_from_list(input_file_names : List[str],
                        output_file_name : str,
                        map_file_name    : str,
                        input_path       : str,
                        output_path      : str,
                        map_path         : str)->Tuple[List[str], str, str]:

    path  = input_path
    opath = output_path
    mpath = map_path

    input_dst_filenames = [os.path.expandvars(f"{path}/{file_name}") for file_name in input_file_names]
    output_dst_filename = os.path.expandvars(f"{opath}/{output_file_name}")
    map_filename        = os.path.expandvars(f"{mpath}/{map_file_name}")

    return input_dst_filenames, output_dst_filename, map_filename


def write_monitor_vars(mdf : Series, log_filename : str):
    mdf.to_hdf(log_filename,
              key     = "LOG"  , mode         = "w",
              format  = "table", data_columns = True,
              complib = "zlib" , complevel    = 4)
