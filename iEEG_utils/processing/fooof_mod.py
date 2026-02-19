# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6, 2025
edits feb 2026

@author: jmile3
"""

import numpy as np
from scipy import interpolate
from scipy.optimize import curve_fit

# assuming in an environment that has fooof (as fooof, not specparam)
from fooof import FOOOFGroup
from fooof.core.funcs import lorentzian_function
# from fooof.utils import interpolate_spectra
from fooof.utils.params import compute_time_constant, compute_knee_frequency
from fooof.sim.gen import gen_periodic

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
        For initializing a FOOOF object. The default is 0.
    min_peak_height : TYPE, optional
        For peak finding. The default is 0.15.
    peak_threshold : TYPE, optional
        For initializing FOOOF object. The default is 2.

    Returns
    -------
    fg : fitted FOOOFGroup object, can be queried for aperiodic params.

    '''
    fg = FOOOFGroup(peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks,
                    aperiodic_mode=aperiodic_mode,
                    regularization_weight=regularization_weight,
                    min_peak_height=min_peak_height,
                    peak_threshold=peak_threshold)

    fg._ap_guess = ap_guesses
    fg._ap_bounds = ap_bounds

    fg.fit(freqs, spectra, freq_range=freq_range,n_jobs=n_jobs) 
    return fg
    
def model_spect(index,model_obj,freqs,intfs,save_fit=True):
    '''
    index = int (pass to starmap as iterable of ints for multiprocessing)
    model_obj = fooof model object that has been initialized (w/ fit_group)
    freqs = iterable of frequencies to fit and pass to model_obj
    intfs = interpolated pseudo-logarithmic re-spacing of low-frequencies
    save_fit = True saves residual trace; False saves aperiodic parameters 

    returns either a list with:
        residual amplitude of fit at min freq (good fits are ~0)
        INT (from knee frequency)
        offset of aperiodic model
        knee frequency of aperiodic model
        exponent of aperiodic model
        
    OR an array of whitened (residual) power after fitting lorentzian 1/f
    
    NOTE: probably makes more sense to default to list of aperiodic params,
          as they can be used to create the lorentzian and, thus the 
          residual power plot
    ''' 
    # refit the original peak-removed spectrum with reweighted low and high frequency spacing
    pk_rm = model_obj.power_spectra[index]-gen_periodic(freqs,np.ndarray.flatten(model_obj.group_results[index].gaussian_params))
    intfx = interpolate.interp1d(freqs,10**pk_rm)
    offset, knee, exp = model_obj.get_params("aperiodic_params")[index,:]
    try:
        reparams,_ = curve_fit(lorentzian_function, intfs, np.log10(intfx(intfs)), p0=[offset,knee,exp], maxfev=2500)
    except:
        reparams = [offset, knee, exp]
        print("!")
    offset,knee,exp = reparams
    # whitened spectrum (lorentzian corrected)
    resid_trace = model_obj.power_spectra[index]-lorentzian_function(freqs, offset, knee, exp)
    
    # which to return (default is the whitened spectrum)
    if save_fit:
        return resid_trace
    else:
        # in other work, have returned list of aperiodic params to add to df
        # convert knee to INT (in milliseconds, using unlogged knee frequency)
        INT = 1000*compute_time_constant(10**(compute_knee_frequency(knee, exp)))
        return [resid_trace[0],INT,offset,knee,exp]
    