"""


"""


#---- imports

import os
import time
import datetime

import tables            as tb
import numpy             as np

import invisible_cities.core.fit_functions as fitf

from invisible_cities.core .core_functions import in_range

from krcal.dev.table_info import RunInfo
from krcal.dev.table_info import MapInfo

import kr.dev.akr_functions     as akr
import kr.utils.kdst_functions  as kdstf
import kr.dev.corrections       as corrections

from mygst_funtions import Vaxis

#----  configuration

run_number                 = 6267
input_dst_filenames        = [f"$IC_DATA/dsts/kdst_{run_number}_0_filtered.h5"]
#                              f"$IC_DATA/RunIV/kdst_{run_number}_1_filtered.h5",
#                              f"$IC_DATA/RunIV/kdst_{run_number}_2_filtered.h5",
#                              f"$IC_DATA/RunIV/kdst_{run_number}_3_filtered.h5"
#                             ]

Rrange  =    0., 200.
Zrange  =    0., 550.
XYrange = -200., 200.
E0range = 7.5e3, 13.5e3
LTrange = 1.5e3,  3.0e3

XYnbins      =  100


#----  input filter_exits

input_dst_filenames = [os.path.expandvars(fi) for fi in input_dst_filenames]

if multiple_kdsts:
    dst = kdstf.load_dsts(input_dst_filenames, "DST", "Events")
else:
    dst = dstf.load_dst(input_dst_filename, "DST", "Events")

unique_events = ~dst.event.duplicated()

number_of_S2s_full  = np.size         (unique_events)
number_of_evts_full = np.count_nonzero(unique_events)

print(f"Total number of S2s   : {number_of_S2s_full} ")
print(f"Total number of events: {number_of_evts_full}")


#---- relevant data


X   = dst.X   .values
Y   = dst.Y   .values
Z   = dst.Z   .values
R   = dst.R   .values
Phi = dst.Phi .values
S2e = dst.S2e .values
S2q = dst.S2q .values
T   = dst.time.values
TH  = (T - T.min())/3600. # time in hours


#--- selection


sel_r = in_range(R, *Rrange)
sel_z = in_range(Z, *Zrange)
sel   = sel_r & sel_z


#---  number of Events

nevt, *_ = hst.histogram2d(X[sel], Y[sel], (XYa.bins, XYa.bins))

if (np.mean(nevt) < min_number_events):
    assert('Not enough events to produce the lifetime map')


#---- lifetime fits

XYa = hst.Vaxis( XYrange, nbins = XYnbins)

xye0, xylt, xychi2, xyok = akr.ltmap_lsqfit(X[sel], Y[sel], Z[sel], S2e[sel], XYa.bins)

xyq0, xyqlt, xyqchi2, xyqok = akr.ltmap_lsqfit(X[sel], Y[sel], Z[sel], S2q[sel], XYa.bins)

#---- store into the correction files

Trange = (T.min(), T.max())
corrections.write_lifetime_correction(run_number, Trange, XYa.bins,
                                      xye0, xylt, xyok, xyq0, xyqlt, xyqok, nevt)

print('Done')
