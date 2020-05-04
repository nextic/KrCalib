import pandas as pd
import numpy  as np
import datetime
from . io_functions import plot_and_save_evolution_figure

def par_selection_to_plot_vs_time(evol_table: pd.DataFrame,
                                  file_name : str         ):
    """
    Selects parameters in time_evolution map table and
    applies over them a function to plot all of them
    Parameters
    ----------
    evol_table : pd.DataFrame
        Table with temporal evolution information.
    file_name : string
        Standard name for saved files.

    Returns
    ----------
    Nothing
    """
    time_      = list(map(datetime.datetime.fromtimestamp, evol_table.ts))
    units_vect = ['pes', 'mus', 'mm/mus', '%', 'ns', 'pes',
                  'pes', 'mus', 'pes', 'pes', 'pes', '# SiPM']

    for idx, par in enumerate(evol_table.columns[1:-7:2]):
        plot_and_save_evolution_figure(time         = time_              ,
                                       param_name   = par                ,
                                       param        = evol_table[par]    ,
                                       param_u      = evol_table[par+'u'],
                                       units        = units_vect[idx]    ,
                                       file_name    = file_name          ,
                                       n_sigmas_lim = 5             )
    for idx, par in enumerate(evol_table.columns[-3:]):
        plot_and_save_evolution_figure(time         = time_          ,
                                       param_name   = par            ,
                                       param        = evol_table[par],
                                       param_u      = 0              ,
                                       units        = '%'            ,
                                       file_name    = file_name      ,
                                       n_sigmas_lim = 10             )
    return
