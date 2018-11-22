import os
import time
import numpy             as np
import collections       as collections
import pandas            as pd

from   invisible_cities.io.dst_io  import load_dst
import krcal.dev.corrections       as corrections
import csth .utils.cepeak          as cpk
from   csth .utils.cepeak          import EPeak, ESum, CepkTable

Q0MIN  = 6.


#---------------------------------------------
#  Driver function
#---------------------------------------------


def esum(input_filename, output_filename, correction_filename,
            run_number, location,
            q0min = Q0MIN, full = False):
    """ produces an h5 output file with an ntuple with summary information per event-peak.
        if full = True it stores also the corrected event-peak information.
        Parameters:
            input_filename  : (str) input file name  (h5 pmaps file)
            output_filename : (str) output file name (h5 with the )
            correction_filename : (str) correction file name (with the xy and lifetime maps)
            run_number      : (int) run number
            location        : (int) index file (of pmaps)
            q0min           : (float) minimum charge to consider a SiPM hit per slice (default 6.)
            full            : (bool) if full is True the information of the corrected event-peaks is stored in the
                                     output h5 file
        Returns:
            counters        : (int, int) number of total and accepted event-peaks
            data            : (esum, cepks) output data, esum is a DF with the event summary information,
                               cepks are 3 DFs with the information of the corrected event-peaks (event, slices, hits)
                               cepks is only returned if full is True
    """

    try:
        hits = data(input_filename)
    except:
        return (0, 0), None

    calibrate = tools(correction_filename, run_number)

    ntotal       = nepeaks(hits)
    esums, cepks = tables(hits, q0min = q0min, full)

    for iloc, ehits in epeak_iterator(hits):
        evt, ipk = iloc

        epk   = epeak(ehits, q0min)
        if (epk is None): continue

        cepk  = cpk.cepeak(epk, calibrate)
        if (full):
            cepks.set(cepk, iloc)

        timestamp = np.unique(ehits.time)[0]

        esum       = cpk.esum(cepk, location, 0., 0., timestamp)

        esums.set(esum, iloc)

    esums.to_hdf(output_filename)
    if (full):
        cepks.to_hdf(output_filename)

    naccepted = len(esums)
    counter s = (ntotal, naccepted)

    odata    = esum.df() if not full else (esum.df(), cepks.df())

    return counters, odata


def tools(correction_filename, run_number):

    calibrate  = corrections.Calibration(correction_filename, 'scale')

    return calibrate


def data(input_filename):

    try:
        hits = load_dst(input_filename, 'RECO', 'Events')
    except:
        print('Not able to load file : ', input_filename)
        raise IOError

    print('processing ', input_filename)

    return hits


def tables(hits, q0min = Q0MIN):

    nepks  = nepeaks(hits)
    esums  = cpk.ESum(nepks)

    nslices = len(hits.Q)
    nhits   = np.sum(hits.Q > q0min)
    cepks   = CepkTable(nepks, nslices, nhits)
    return esums, cepks


#-----------------------------
#   Main functions
#-----------------------------

def nepeaks(hits):

    groups = hits.groupby(['event', 'npeak'])
    return len(groups)


def epeak_iterator(hits):

    groups = hits.groupby(['event', 'npeak'])
    return iter(groups)


def epeak(hits, q0min = Q0MIN):
    """ returns information form the event-peak (per slices and hits)
    inputs:
        pmap  : (DFPmap)    pmap info per a given event-peak
        q0min : (float)
    returns:
        q0    : (float) total charge in the event-peak
        n0hits: (int)   total number of SiPMs with a signal > 0
        e0i   : (np.array, size = nslices) energy of the slices
        zi    : (np.array, size = nslices) z-position of the slices
        xij, yij, zij : (np.array, size = nhits) x, y, z posisition of the SiPMs with charge > q0min
        q0ij  : (np.array, size = nhits) charge of the SiPMs with charge > q0min
    """

    q0                         = np.sum(hits.Q [hits.Q > 0])

    nslices, e0i, zi           = _slices(hits)
    if (nslices < 0): return None

    nhits, xij, yij, zij, q0ij = _hits(hits, q0min)
    if (nhits <= 0):  return None

    return EPeak(nslices, nhits, q0, e0i, zi, xij, yij, zij, q0ij)


#---------------------------


def _slices(hits):

    zij    = hits.Z.values
    zi     = np.unique(zij)
    nslices = len(zi)
    if (nslices <= 0):
        return nslices, None, None

    e0ij      = hits.E.values
    selslices = [zij == izi for izi in zi]
    e0i  = np.array([np.sum(e0ij[sel]) for sel in selslices])

    #print('nslices, t0, e0i, zi', nslices, t0, e0i, zi)
    return nslices, e0i, zi


def _hits(hits, q0min = Q0MIN):

    qsel    = hits.Q > q0min
    nhits   = np.sum(qsel)
    if (nhits <= 0):
            return nhits, None, None, None, None

    q0ij   = hits.Q [qsel].values
    xij    = hits.X [qsel].values
    yij    = hits.Y [qsel].values
    zij    = hits.Z [qsel].values

    #print('nhits, xij, yij, zij, q0ij ', nhits, len(xij), len(yij), len(zij), len(q0ij))
    #print('xij, yij, zij, q0ij ', xij, yij, zij, q0ij)
    return nhits, xij, yij, zij, q0ij
