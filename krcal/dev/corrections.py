#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri 6 Jul 2018

@author: hernando
"""

import os
import tables            as tb
import numpy             as np

import invisible_cities.reco.dst_functions as dstf
import invisible_cities.io  .kdst_io       as kdstio

from krcal.dev.table_info import RunInfo
from krcal.dev.table_info import MapInfo

from collections import namedtuple

default_calibration_filename = f"$IC_DATA/maps/akr_corrections_run6206.h5"
default_calibration_filename = os.path.expandvars(default_calibration_filename)

XYMap = namedtuple('XYMap',
                   ('x', 'y', 'values', 'errors', 'valid'))


class Calibration:

    def __init__(self, correction_filename = default_calibration_filename,
                        node = 'geometry'):
        """ class that holds the calibration functions.
        node: (str) 'scale' or 'geometry' for lifetime scale or geometry map
        call() to apply corrections.
        """

        print('calibration file :', correction_filename)
        print('node             :', node)

        self.E0_correction = dstf.load_xy_corrections(correction_filename,
                                    group = "XYcorrections",
                                    node  = "E"+node,
                                    norm_strategy =  "index",
                                    norm_opts     = {"index": (40, 40)})

        self.ELT_correction  = dstf.load_lifetime_xy_corrections(correction_filename,
                                    group = "XYcorrections",
                                    node  = "Elifetime")

        self.Q0_correction   = dstf.load_xy_corrections(correction_filename,
                                    group = "XYcorrections",
                                    node  = "Q"+node,
                                    norm_strategy =  "index",
                                    norm_opts     = {"index": (40, 40)})

        self.QLT_correction  =  dstf.load_lifetime_xy_corrections(correction_filename,
                                    group = "XYcorrections",
                                    node  = "Qlifetime")

    def __call__(self, X, Y, Z, T, S2e, S2q):
        """ apply lifetime and geometry calibrations
            X, Y, Z, T: np.arrays with the x, y, z, t values
            S2e, S2q  : np.arrays with the S2 PMT (S2e) and SiPM (S2q) signals
        returns:
            E, Q      : np.arrays with the energy (PMTs) and charge (SiPM) corrected signals
        options:
            Z  : None, then only geometrical corrections are applied
            S2q: None, then only E corrected, Q will be None
        """

        E  = S2e * self.E0_correction(X, Y).value
        Q  = None

        if (S2q is not None):
            Q  = S2q * self.Q0_correction(X, Y).value

        if (Z is None):
            return E, Q

        E = E * self.ELT_correction(Z, X, Y).value

        if (S2q is not None):
            Q = Q * self.QLT_correction(Z, X, Y).value

        return E, Q


def get_xymap(correction_filename, xymap_name):
    # xymap_name = 'Escale', 'Elifetime', 'Qscale', 'Qlifetime', 'Egeometry', 'Qgeometry'
    xymap = dstf.load_dst(correction_filename,
                          group = "XYcorrections",
                          node  = xymap_name)

    x   = np.unique(xymap.x.values)
    y   = np.unique(xymap.y.values)
    values = xymap.factor     .values.reshape(x.size, y.size)
    errors = xymap.uncertainty.values.reshape(x.size, y.size)

    sel = values > 0
    errors[sel]  = errors[sel]*100/values[sel]
    errors[~sel] = 0.

    return XYMap(x, y, values, errors, None)


def write_lifetime_correction(correction_filename, run_number, Trange, XYbins,
                              Escale, ELT,  Eok, Qscale, QLT, Qok, nevt,
                              filename_prefix = 'akr_corrections_run'):

    XYnbins   = len(XYbins)-1
    XYrange   = XYbins.min(), XYbins.max()
    XYpitch   = np.diff(XYbins)[0]
    XYcenters = 0.5*(XYbins[:-1]+XYbins[1:])

    print('writing corrections :', correction_filename)
    t_min, t_max = Trange
    with tb.open_file(correction_filename, "w") as correction_file:
        run_table = correction_file.create_table(correction_file.root, "RunInfo"  , RunInfo, "Run metadata")
        map_table = correction_file.create_table(correction_file.root, "LTMapInfo", MapInfo, "Map metadata")

        row = run_table.row
        row["run_number"] = run_number
        row["t_min"     ] = t_min
        row["t_max"     ] = t_max
        row.append()

        row = map_table.row
        row["x_nbins"] = XYnbins
        row["y_nbins"] = XYnbins
        row["x_pitch"] = XYpitch
        row["y_pitch"] = XYpitch
        row["x_min"  ] = XYrange[0]
        row["x_max"  ] = XYrange[1]
        row["y_min"  ] = XYrange[0]
        row["y_max"  ] = XYrange[1]
        row.append()

    e0, e0u = np.mean(Escale.value[Eok]), np.mean(Escale.uncertainty[Eok])
    Escale_safe  = np.where(Eok, Escale.value      , e0)
    Escaleu_safe = np.where(Eok, Escale.uncertainty, e0u)

    lt, ltu = np.mean(ELT.value[Eok]), np.mean(ELT.uncertainty[Eok])
    ELT_safe     = np.where(Eok, ELT.value      ,  lt)
    ELTu_safe    = np.where(Eok, ELT.uncertainty,  ltu)

    q0, q0u = np.mean(Qscale.value[Qok]), np.mean(Qscale.uncertainty[Qok])
    Qscale_safe  = np.where(Qok, Qscale.value      , q0)
    Qscaleu_safe = np.where(Qok, Qscale.uncertainty, q0u)

    qlt, qltu = np.mean(QLT.value[Qok]), np.mean(QLT.uncertainty[Qok])
    QLT_safe     = np.where(Qok, QLT.value      ,  qlt)
    QLTu_safe    = np.where(Qok, QLT.uncertainty,  qltu)

    with tb.open_file(correction_filename, "a") as correction_file:
        write_escale = kdstio.xy_writer(correction_file,
                                        group       = "XYcorrections",
                                        name        = "Escale",
                                        description = "XY-dependent energy scale",
                                        compression = "ZLIB4")
        write_escale(XYcenters, XYcenters, Escale_safe, Escaleu_safe, nevt)

        write_lifetime = kdstio.xy_writer(correction_file,
                                        group       = "XYcorrections",
                                        name        = "Elifetime",
                                        description = "XY-dependent lifetime values",
                                        compression = "ZLIB4")
        write_lifetime(XYcenters, XYcenters, ELT_safe, ELTu_safe, nevt)

        write_qscale = kdstio.xy_writer(correction_file,
                                        group       = "XYcorrections",
                                        name        = "Qscale",
                                        description = "XY-dependent energy scale",
                                        compression = "ZLIB4")
        write_qscale(XYcenters, XYcenters, Qscale_safe, Qscaleu_safe, nevt)

        write_qlifetime = kdstio.xy_writer(correction_file,
                                        group       = "XYcorrections",
                                        name        = "Qlifetime",
                                        description = "XY-dependent lifetime values",
                                        compression = "ZLIB4")
        write_qlifetime(XYcenters, XYcenters, QLT_safe, QLTu_safe, nevt)
