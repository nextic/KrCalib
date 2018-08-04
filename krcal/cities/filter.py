"""
   This SCRIP runs the filtering of a list of kdsts
   selects 1S1 and 1S2 in nsigma (3.5) around the Kr E vs Z band
   it uses a boostrap calibration
   it writes a filtered dst
"""

#---- imports

import os
import time
import glob

import tables            as tb
import numpy             as np

import invisible_cities.reco.dst_functions as dstf
from invisible_cities.core .core_functions import in_range

import krcal.dev.corrections       as corrections
import krcal.dev.akr_functions     as akr
import krcal.utils.kdst_functions  as kdstf

#--- Configuration

run_number            = 6267
path                  = f"$IC_DATA/"
tag                   = 'trigger1_v0.9.9_20180802'
subrun                = 0
file_range            = (10000*subrun, 10000*(subrun+1))

run_number_correction = 6206
correction_filename   = f"$IC_DATA/maps/corrections_run{run_number_correction}.h5"
write_filtered_dst    = True
apply_geocorrection   = True
selection_1s2         = True

# ----  selection parameters

S2range = (2e3  , 20e3)  # this is a selection
nsigma  = 3.5            # sigma for the E vs Z band selection

#-------- Input dataclass

input_dst_filenames = kdstf.kdst_filenames_in_file_range(path, run_number, tag, file_range)


#-------- Filter data

sel_1S1 = dst.nS1 == 1
kdstf.selection_info(sel_1S1, 'one S1')
dst = dst[sel_1S1]

sel_1S2 = dst.nS2 == 1
kdstf.selection_info(sel_1S2, 'one S2')
dst = dst[sel_1S2]


number_of_S2s_full, number_of_evts_full = kdstf.kdst_unique_events(dst)

print(f"Total number of S2s   : {number_of_S2s_full} ")
print(f"Total number of events: {number_of_evts_full}")


#--- temporaly fix (from Gonzalo's)
if "index" in dst: del dst["index"]

#--- relevant variables

X   = dst.X   .values
Y   = dst.Y   .values
Z   = dst.Z   .values
T   = dst.time.values
S2e = dst.S2e .values
S2q = dst.S2q .values
S1e = dst.S1e .values
TH = (T - T.min())/3600. # time in hours

#---- bootstrap calibrattion

E = S2e
Q = S2q

if (apply_geocorrection):
    correction_filename = os.path.expandvars(correction_filename)
    calibrate = corrections.Calibration(correction_filename)
    E, Q = calibrate(X, Y, Z=None, T, S2e, S2q)

#--- filtering

sel_nsipm = dst.Nsipm > 1
kdstf.selection_info(sel_nsipm, '#Sipm >1 ')
sel_S2e   = in_range(S2e,  *S2range)
kdstf.selection_info(sel_S2e, 'S2e range ' )

sel = sel_nsipm & sel_S2e
kdstf.selection_info(sel)


#--- selectin in the band

Znbins, Zrange = 100, (0., 550.)
Zfitrange      = (50., 500.)
Erange         = (4e3, 16e3)


sel_EvsZ = akr.selection_in_band(E, Z, Erange, Zrange, Zfitrange, nsigma = nsigma, plot=False);
kdstf.selection_info(sel_EvsZ, 'in E vs Z band ')
sel = sel & sel_EvsZ
kdstf.selection_info(sel, 'total selection')

#--  write filter dst2

if (write_filtered_dst):
    output_dst_filename = f"{path}/dsts/kdst_{run_number}_{subrun}_filtered.h5"
    output_dst_filename = os.path.expandvars(output_dst_filename)
    print('writing filtered dst ', output_dst_filename)
    kdstf.kdst_write(dst[sel], output_dst_filename);

print('Done!')
