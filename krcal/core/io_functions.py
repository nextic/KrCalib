import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates  as md
from   invisible_cities.core.core_functions import shift_to_bin_centers
from   typing         import Tuple
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


def compute_and_save_hist_as_pd(values     : np.array           ,
                                out_file   : pd.HDFStore        ,
                                hist_name  : str                ,
                                n_bins     : int                ,
                                range_hist : Tuple[float, float],
                                norm       : bool = False       )->None:
    """
    Computes 1d-histogram and saves it in a file.
    The name of the table inside the file must be provided.
    Parameters
    ----------
    values : np.array
        Array with values to be plotted.
    out_file: pd.HDFStore
        File where histogram will be saved.
    hist_name: string
        Name of the pd.Dataframe to contain the histogram.
    n_bins: int
        Number of bins to make the histogram.
    range_hist: length-2 tuple (optional)
        Range of the histogram.
    norm: bool
        If True, histogram will be normalized.
    """
    n, b = np.histogram(values, bins = n_bins,
                        range = range_hist,
                        density = norm)
    table = pd.DataFrame({'entries': n,
                          'magnitude': shift_to_bin_centers(b)})
    out_file.put(hist_name, table, format='table', data_columns=True)

    return



def compute_and_save_hist_as_pdf(values     : np.ndarray          ,
                                 out_file   : str                ,
                                 n_bins     : int                ,
                                 range_hist : Tuple[float, float],
                                 title      : str                ,
                                 x_label    : str                ,
                                 y_range    : Tuple[float, float],
                                 norm       : bool = False       )->None:
    """
    Computes 1d-histogram and saves it as a pdf image.
    Parameters
    ----------
    values : np.ndarray
        Array with values to be plotted.
    out_file: string
        File where histogram will be saved.
    n_bins: int
        Number of bins to make the histogram.
    range_hist: length-2 tuple (optional)
        Range of the histogram.
    title: str
        Title for the plot.
    x_label: str
        Label for X-axis.
    y_range: length-2 tuple
        Limit fot Y-axis.
    norm: bool
        If True, histogram will be normalized.
    """
    fig = plt.figure()
    plt.hist(values                ,
             bins      = n_bins    ,
             range     = range_hist,
             density   = norm      ,
             histtype  = 'step'    ,
             linewidth = 2         );
    plt.ylabel('Entries')
    plt.xlabel(x_label)
    plt.title(title)
    plt.ylim(y_range)
    plt.grid(True, alpha=0.5, color='k', linestyle=':')
    fig.savefig(out_file.format(title).replace(" ", ""), bbox_inches='tight')

    return


def plot_and_save_evolution_figure(time        : np.ndarray  ,
                                   param_name  : str        ,
                                   param       : np.ndarray  ,
                                   param_u     : np.ndarray  ,
                                   units       : str        ,
                                   file_name   : str        ,
                                   n_sigmas_lim: int     = 1):
    """
    Plot and save evolution of a given parameter vs time.
    Parameters
    ----------
    time : np.ndarray
        Array with the centers of each time interval (X-values).
    param_name : string
        Name of the parameter to plot.
    param : np.ndarray
        Array with the magnitude to plot (Y-values).
    param_u : np.ndarray
        Array with uncertainty values.
    units : string
        Units to be shown in x-label.
    file_name : string
        Name of file for plots. It must contain '{0}' to be name after
        the plotted parameter.
    n_sigmas_lim : int (optional)
        Number of sigmas to set the Y-axis limits in plots.

    Returns
    ----------
    Nothing
    """
    fig = plt.figure()
    plt.errorbar(time, param, param_u, fmt='.')
    plt.xlabel('Date')
    plt.ylabel('{0} ({1})'.format(param_name, units))
    plt.title('{0}'.format(param_name))
    mean = np.mean(param)
    std  = np.std(param)
    plt.ylim(param.min()-std*n_sigmas_lim,
             param.max()+std*n_sigmas_lim)
    plt.grid(True, alpha=0.5, color='k', linestyle=':')
    ax   = plt.gca()
    xfmt = md.DateFormatter('%d-%m %H:%M')
    ax.xaxis.set_major_formatter(xfmt)
    plt.xticks( rotation=25 )
    plt.text(0.03, 0.9,
             'Mean= {0} \n Std= {1}'.format(np.round(mean,4), np.round(std,4)),
             fontsize  = 10,
             transform = ax.transAxes,
             bbox      = {'facecolor': 'white', 'alpha': 1, 'pad': 5})
    fig.savefig(file_name.format(param_name), bbox_inches='tight')

    return
