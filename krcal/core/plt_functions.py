import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . histo_functions import labels
from . histo_functions import h1, plot_histo
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


def plot_energy_vs_t(kce, krBins, krRanges, figsize=(14,10)):


    fig = plt.figure(figsize=figsize)
    fig.add_subplot(2, 2, 1)
    (_) = h1(kce.T, bins = krBins.T, range =krRanges.T)

    fig.add_subplot(2, 2, 2)
    nevt, *_  = plt.hist2d(kce.T, kce.E, (krBins.T, krBins.S2e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("T", "E (pes)", f" E (corrected) vs T"))

    fig.add_subplot(2, 2, 3)
    nevt, *_  = plt.hist2d(kce.T, kce.Q, (krBins.T, krBins.S2q))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("T", "Q (pes)", f" Q (corrected)  vs T"))

    fig.add_subplot(2, 2, 4)
    nevt, *_  = plt.hist2d(kce.T, kce.S1e, (krBins.T, krBins.S1e))
    plt.colorbar().set_label("Number of events")
    labels(PlotLabels("T", "S2q (pes)", f" S1 vs T"))
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

def plot_energy_vs_z_histo_and_profile(e, zc, emean_z, esigma_z,
                                       bins_e = 50, range_e = (2e+3,14e+3),
                                       figsize=(10,6)):
    fig = plt.figure(figsize=figsize)
    pltLabels =PlotLabels(x='Energy-like', y='Events', title='true')
    ax      = fig.add_subplot(1, 2, 1)
    (_)     = h1(e, bins=50, range=(7e+3,11e+3))
    plot_histo(pltLabels, ax)

    ax      = fig.add_subplot(1, 2, 2)
    plt.errorbar(zc, emean_z, esigma_z, np.diff(zc)[0]/2, fmt="kp", ms=7, lw=3)

    plt.tight_layout()
