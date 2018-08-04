#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 2018

Plots functions for Kr,

@author: G. Martinez, J.A.hernando
"""

import os

import numpy              as np
import scipy.stats        as stats
import tables             as tb
import pandas             as pd


import tables as tb
from tables import NoSuchNodeError
from tables import HDF5ExtError
import warnings


#---------- load dsts

def load_dst(filename, group, node):
    try:
        with tb.open_file(filename) as h5in:
            try:
                table = getattr(getattr(h5in.root, group), node).read()
                return pd.DataFrame.from_records(table)
            except NoSuchNodeError:
                print(f' warning:  {filename} not of kdst type')
    except HDF5ExtError:
        print(f' warning:  {filename} corrupted')

def load_dsts(dst_list, group, node):
    dsts = [load_dst(filename, group, node) for filename in dst_list]
    return pd.concat(dsts)


def kdst_unique_events(dst):
    unique_events = ~dst.event.duplicated()

    number_of_S2s_full  = np.size         (unique_events)
    number_of_evts_full = np.count_nonzero(unique_events)


    return number_of_S2s_full, number_of_evts_full


#---- write dists

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

#----- for reading up multiple kdsts

def _numbers_from_file_range(file_range):
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

def kdst_filenames_in_file_range(path, run_number, tag, file_range, filter_exits=True):
    """ return the full filename of a run an tag in a file_rage (tuple with 2 entries)
    If filter_exits, returns only the files that are in the path directory
    """
    N = _numbers_from_file_range(file_range)
    cfs = [os.path.expandvars(f"{path}/{run_number}/kdst_{number}_{run_number}_{tag}_krth.h5") for number in N]

    if (filter_exits == False):
        return cfs

    fs = os.listdir(os.path.expandvars(f"{path}/{run_number}/"))
    fs = [fi for fi in fs if fi.find('.h5')>0]
    fs.sort()
    complete_path = os.path.expandvars(f"{path}/{run_number}/")
    fs = [complete_path + fi for fi in fs]

    input_dst_filenames   = [fi for fi in cfs if fi in fs]
    missing_dst_filenames = [fi for fi in cfs if fi not in fs]
    print('missing files ', len(missing_dst_filenames))

    if (len(missing_dst_filenames)>0):
        def get_number(fi):
            words = fi.split('/')
            words = words[-1].split('_')
            return words[1]
        numbers = [get_number(fi) for fi in missing_dst_filenames]
        print('missing files ', numbers)

    return input_dst_filenames

#----  seleccion useful functions

def selection_info(sel, comment=''):
    nsel   = np.sum(sel)
    effsel = 100.*nsel/(1.*len(sel))
    s = f"Total number of selected candidates {comment}: {nsel} ({effsel:.1f} %)"
    print(s)
    return s
