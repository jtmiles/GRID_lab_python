# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 15:44:30 2025

@author: jmile3
"""

# assuming in an environment that has fooof (as fooof, not specparam)
from fooof import FOOOFGroup

def fit_group(freqs, spectra, freq_range = [0.5,100], n_jobs = -1,
              ap_guesses = (-1.5,1,2.5), 
              ap_bounds = [(-3.5,0.5), (-2, 3), (0.5, 5)],
              peak_width_limits=[1, 40],max_n_peaks=6,aperiodic_mode='lorentzian',
              regularization_weight=0,min_peak_height=0.15,peak_threshold=2):
    '''
    
    NOTE:
        this is just to try and speed up some fooof operations, which seem to
        be very fast on CPU but slow wall time, perhaps b/c of ipython usage
        
    Parameters
    ----------
    freqs : np.array
        frequency values for spectra
    spectra : np.array
        arrays to fit, each row is a spectrum (col is frequency)
    freq_range : list
        list of [min,max] frequencies to fit (in Hz)
    n_jobs : int
        number of cores to use for processing, -1 is all available
    ap_guesses : tuple, optional
        guesses at curve fitting parameters
        assuming "lorentzian" mode, defaults are (-1.5,1,2.5)
    ap_bounds : list of tuples
        bounds for guesses (optimization must return parameters w/in bounds)
    peak_width_limits : list, optional
        list of [min,max] peak widths (Hz); The default is [1, 40].
    max_n_peaks : int, optional
        max number of peaks; The default is 6.
    aperiodic_mode : TYPE, optional
        type of aperiodic model; The default is 'lorentzian'.
    regularization_weight : TYPE, optional
        DESCRIPTION. The default is 0.
    min_peak_height : TYPE, optional
        DESCRIPTION. The default is 0.15.
    peak_threshold : TYPE, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    fgs : fitted FOOOFGroup object, can be queried for aperiodic params.

    '''
    fg = FOOOFGroup(peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks,
                    aperiodic_mode=aperiodic_mode,
                    regularization_weight=regularization_weight,
                    min_peak_height=min_peak_height,
                    peak_threshold=peak_threshold)

    fg._ap_guess = ap_guesses
    fg._ap_bounds = ap_bounds

    fgs = fg.fit(freqs, spectra, freq_range=freq_range,n_jobs=n_jobs) 
    return fgs
    