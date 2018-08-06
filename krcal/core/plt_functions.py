import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . histo_functions import labels
from . kr_types        import PlotLabels
# import matplotlib.dates  as md
# from   invisible_cities.icaro.mpl_functions import set_plot_labels
# from   invisible_cities.core.system_of_units_c import units
#
#
# from   invisible_cities.evm  .ic_containers  import Measurement
# from   invisible_cities.icaro. hst_functions import display_matrix


# def figsize(type="small"):
#     if type == "S":
#         plt.rcParams["figure.figsize"]  = 8, 6
#     elif type == "s":
#          plt.rcParams["figure.figsize"] = 6, 4
#     elif type == "l":
#         plt.rcParams["figure.figsize"] = 10, 8
#     else:
#         plt.rcParams["figure.figsize"] = 12, 10

def plot_xy_density(dst, xybins, figsize=(10,8)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)
    XYpitch = np.diff(xybins)[0]
    nevt_full, *_ = plt.hist2d(dst.X, dst.Y, (xybins, xybins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("X (mm)", "Y (mm)", f"full distribution for {XYpitch:.1f} mm pitch"))
    return nevt_full


def plot_s1_vs_z(dst, zbins, s1bins, figsize=(10,8)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)
    nevt, *_  = plt.hist2d(dst.Z, dst.S1e, (zbins, s1bins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S1 (pes)", f"S1 vs Z"))


def plot_s2_vs_z(dst, zbins, s2bins, figsize=(10,8)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)
    nevt, *_  = plt.hist2d(dst.Z, dst.S2e, (zbins, s2bins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2 (pes)", f"S2 vs Z"))


def plot_s2_vs_s1(dst, s1bins, s2bins, figsize=(10,8)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)
    nevt, *_  = plt.hist2d(dst.S1e, dst.S2e, (s1bins, s2bins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S1 (pes)", "S2 (pes)", f"S2 vs S1"))


def plot_q_vs_s2(dst, s2bins, qbins, figsize=(10,8)):
    fig = plt.figure(figsize=figsize)
    fig.add_subplot(1, 1, 1)
    nevt, *_  = plt.hist2d(dst.S2e, dst.S2q, (s2bins, qbins))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S2 (pes)", "Q (pes)", f"Q vs S2"))


def plot_s2e_vs_z_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.Z, full.S2e, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2e (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.Z, fid.S2e, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2e (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.Z, core.S2e, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2e (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.Z, hcore.S2e, (krBins.Z, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2e (pes)", f" hard core Z"))
    plt.tight_layout()


def plot_s1e_vs_z_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.Z, full.S1e, (krBins.Z, krBins.S1e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S1e (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.Z, fid.S1e, (krBins.Z, krBins.S1e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S1e (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.Z, core.S1e, (krBins.Z, krBins.S1e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S1e (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.Z, hcore.S1e, (krBins.Z, krBins.S1e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S1e (pes)", f" hard core Z"))
    plt.tight_layout()


def plot_s2q_vs_z_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.Z, full.S2q, (krBins.Z, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2q (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.Z, fid.S2q, (krBins.Z, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2q (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.Z, core.S2q, (krBins.Z, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2q (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.Z, hcore.S2q, (krBins.Z, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("Z (mm)", "S2q (pes)", f" hard core Z"))
    plt.tight_layout()


def plot_s2e_vs_s1e_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.S1e, full.S2e, (krBins.S1e, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S1e (pes)", "S2e (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.S1e, fid.S2e, (krBins.S1e, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S1e (pes)", "S2e (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.S1e, core.S2e, (krBins.S1e, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S1e (pes)", "S2e (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.S1e, hcore.S2e, (krBins.S1e, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S1e (pes)", "S2e (pes)", f" hard core Z"))
    plt.tight_layout()


def plot_s2q_vs_s2e_r_regions(kdsts, krBins, figsize=(14,10)):

    full, fid, core, hcore = kdsts

    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    nevt, *_  = plt.hist2d(full.S2e, full.S2q, (krBins.S2e, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S2e (pes)", "S2q (pes)", f" full "))

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(fid.S2e, fid.S2q, (krBins.S2e, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S2e (pes)", "S2q (pes)", f" fid "))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(core.S2e, core.S2q, (krBins.S2e, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S2e (pes)", "S2q (pes)", f" core "))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(hcore.S2e, hcore.S2q, (krBins.S2e, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("S2e (pes)", "S2q (pes)", f" hard core Z"))
    plt.tight_layout()



def plot_energy_distributions(dst, zbins, s1bins, s2bins, qbins, figsize=(14,10)):
        fig = plt.figure(figsize=figsize)

        fig.add_subplot(2, 2, 1)
        nevt, *_  = plt.hist2d(dst.Z, dst.S1e, (zbins, s1bins))
        plt.colorbar().set_label("Number of events")
        labels(PlotLabels("Z (mm)", "S1 (pes)", f"S1 vs Z"))

        fig.add_subplot(2, 2, 2)
        nevt, *_  = plt.hist2d(dst.Z, dst.S2e, (zbins, s2bins))
        plt.colorbar().set_label("Number of events")
        labels(PlotLabels("Z (mm)", "S2 (pes)", f"S2 vs Z"))

        fig.add_subplot(2, 2, 3)
        nevt, *_  = plt.hist2d(dst.S1e, dst.S2e, (s1bins, s2bins))
        plt.colorbar().set_label("Number of events")
        labels(PlotLabels("S1 (pes)", "S2 (pes)", f"S2 vs S1"))

        fig.add_subplot(2, 2, 4)
        nevt, *_  = plt.hist2d(dst.S2e, dst.S2q, (s2bins, qbins))
        plt.colorbar().set_label("Number of events")
        labels(PlotLabels("S2 (pes)", "Q (pes)", f"Q vs S2"))

#     fig.add_subplot(2, 2, 2)
#     nevt, *_  = plt.hist2d(kdst.fid.Z, kdst.fid.E, (krBins.Z, krBins.E))
#     plt.colorbar().set_label("Number of events")
#     labels("Z (mm)", "E (pes)", f" fid ")
#
#     fig.add_subplot(2, 2, 3)
#     nevt, *_  = plt.hist2d(kdst.core.Z, kdst.core.E, (krBins.Z, krBins.E))
#     plt.colorbar().set_label("Number of events")
#     labels("Z (mm)", "E (pes)", f" core ")
#
#     fig.add_subplot(2, 2, 4)
#     nevt, *_  = plt.hist2d(kdst.hcore.Z, kdst.hcore.E, (krBins.Z, krBins.E))
#     plt.colorbar().set_label("Number of events")
#     labels("Z (mm)", "E (pes)", f" hard core Z")
#     plt.tight_layout()
#
#
# def plot_s1_vs_z(kdst, krBins, figsize=(14,10)):
#     fig = plt.figure(figsize=figsize)
#     fig.add_subplot(2, 2, 1)
#     nevt, *_  = plt.hist2d(kdst.full.Z, kdst.full.S1, (krBins.Z, krBins.S1))
#     plt.colorbar().set_label("Number of events")
#     labels("Z (mm)", "S1 (pes)", f"full S1 vs Z")
#
#     fig.add_subplot(2, 2, 2)
#     nevt, *_  = plt.hist2d(kdst.fid.Z, kdst.fid.S1, (krBins.Z, krBins.S1))
#     plt.colorbar().set_label("Number of events")
#     labels("Z (mm)", "S1 (pes)", f"fid S1 vs Z")
#
#     fig.add_subplot(2, 2, 3)
#     nevt, *_  = plt.hist2d(kdst.core.Z, kdst.core.S1, (krBins.Z, krBins.S1))
#     plt.colorbar().set_label("core Number of events")
#     labels("Z (mm)", "S1 (pes)", f"S1 vs Z")
#
#     fig.add_subplot(2, 2, 4)
#     nevt, *_  = plt.hist2d(kdst.hcore.Z, kdst.hcore.S1, (krBins.Z, krBins.S1))
#     plt.colorbar().set_label("hard core Number of events")
#     labels("Z (mm)", "S1 (pes)", f"S1 vs Z")
#     plt.tight_layout()
#
# def plot_s2_vs_s1(kdst, krBins, figsize=(14,10)):
#     fig = plt.figure(figsize=figsize)
#     fig.add_subplot(2, 2, 1)
#     nevt, *_  = plt.hist2d(kdst.full.S1, kdst.full.E, (krBins.S1, krBins.E))
#     plt.colorbar().set_label("Number of events")
#     labels("S1 (pes)", "S2 (pes)", f"full S2 vs S1")
#
#     fig.add_subplot(2, 2, 2)
#     nevt, *_  = plt.hist2d(kdst.fid.S1, kdst.fid.E, (krBins.S1, krBins.E))
#     plt.colorbar().set_label("Number of events")
#     labels("S1 (pes)", "S2 (pes)", f"fid S2 vs S1")
#     fig.add_subplot(2, 2, 3)
#
#     nevt, *_  = plt.hist2d(kdst.core.S1, kdst.core.E, (krBins.S1, krBins.E))
#     plt.colorbar().set_label("core Number of events")
#     labels("S1 (pes)", "S2 (pes)", f"core S2 vs S1")
#     fig.add_subplot(2, 2, 4)
#
#     nevt, *_  = plt.hist2d(kdst.hcore.S1, kdst.hcore.E, (krBins.S1, krBins.E))
#     plt.colorbar().set_label("hard core Number of events")
#     labels("S1 (pes)", "S2 (pes)", f"hard core S2 vs S1")
#     plt.tight_layout()
#
#
# def plot_lifetime_T(kfs, timeStamps, ltlim=(2000, 3000),  figsize=(12,6)):
#     ez0s = [kf.par[0] for kf in kfs]
#     lts = [np.abs(kf.par[1]) for kf in kfs]
#     u_ez0s = [kf.err[0] for kf in kfs]
#     u_lts = [kf.err[1] for kf in kfs]
#     plt.figure(figsize=figsize)
#     ax=plt.gca()
#     xfmt = md.DateFormatter('%d-%m %H:%M')
#     ax.xaxis.set_major_formatter(xfmt)
#     plt.errorbar(timeStamps, lts, u_lts, fmt="kp", ms=7, lw=3)
#     plt.xlabel('date')
#     plt.ylabel('Lifetime (mus)')
#     plt.ylim(ltlim)
#     plt.xticks( rotation=25 )
#
# def display_lifetime_maps(Escale : Measurement,
#                           ELT: Measurement,
#                           kltl : KrLTLimits,
#                           XYcenters : np.array,
#                           cmap = "jet",
#                           mask = None):
#
#     """Display lifetime maps: the mask allow to specify channels
#     to be masked out (usually bad channels)
#     """
#
#     #fig = plt.figure(figsize=figsize)
#     #fig.add_subplot(2, 2, 1)
#     plt.subplot(2, 2, 1)
#     *_, cb = display_matrix(XYcenters, XYcenters, Escale.value, mask,
#                             vmin = kltl.Es.min,
#                             vmax = kltl.Es.max,
#                             cmap = cmap,
#                             new_figure = False)
#     cb.set_label("Energy scale at z=0 (pes)")
#     labels("X (mm)", "Y (mm)", "Energy scale")
#
#     #fig.add_subplot(2, 2, 2)
#     plt.subplot(2, 2, 2)
#     *_, cb = display_matrix(XYcenters, XYcenters, Escale.uncertainty, mask,
#                         vmin = kltl.Eu.min,
#                         vmax = kltl.Eu.max,
#                         cmap = cmap,
#                         new_figure = False)
#     cb.set_label("Relative energy scale uncertainty (%)")
#     labels("X (mm)", "Y (mm)", "Relative energy scale uncertainty")
#
#     #fig.add_subplot(2, 2, 3)
#     plt.subplot(2, 2, 3)
#     *_, cb = display_matrix(XYcenters, XYcenters, ELT.value, mask,
#                         vmin = kltl.LT.min,
#                         vmax = kltl.LT.max,
#                         cmap = cmap,
#                         new_figure = False)
#     cb.set_label("Lifetime (µs)")
#     labels("X (mm)", "Y (mm)", "Lifetime")
#
#     #fig.add_subplot(2, 2, 4)
#     plt.subplot(2, 2, 4)
#     *_, cb = display_matrix(XYcenters, XYcenters, ELT.uncertainty, mask,
#                         vmin = kltl.LTu.min,
#                         vmax = kltl.LTu.max,
#                         cmap = cmap,
#                         new_figure = False)
#     cb.set_label("Relative lifetime uncertainty (%)")
#     labels("X (mm)", "Y (mm)", "Relative lifetime uncertainty")
#
#     plt.tight_layout()
