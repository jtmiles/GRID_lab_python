# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import median_abs_deviation as MAD
from scipy.interpolate import splev, splrep
from scipy.signal import welch

def rolling_sum(arr, window_size):
    """
    (centered) rolling average from np.convolve with window of ones
    
    Parameters
    ----------
    arr : array (1D)
        array (vector, really) to be summed over.
    window_size : TYPE
        number of points to include in sum.

    Returns
    -------
    array
        moving average over array.
        NOTE: summed values are centered, even at edges.

    """
    kernel = np.ones(window_size, dtype=int)
    return np.convolve(arr, kernel, mode='same')

def bfilt(data, srate, n, fpass, filter_type):
    """
    Butterworth filter for electrophysiological data.

    Parameters:
        data : ndarray
            Raw ephys data (n_samp x n_ch if using multiple channels).
        srate : float
            Sampling rate.
        n : int
            Filter order.
        fpass : list or float
            Frequency bandpass filter.
        filter_type : str
            Filter type ('low', 'high', 'bandpass', or 'stop').

    Returns:
        filt_data : ndarray
            Filtered data.
    """
    # Create Butterworth filter (zeros, poles, gain)
    sos = butter(n, [f / (srate / 2) for f in fpass] if isinstance(fpass, list) else fpass / (srate / 2),
                     btype=filter_type, output="sos")
    
    # Apply the filter using filtfilt to correct phase slips
    filt_data = sosfiltfilt(sos, data, axis=0)
    
    return filt_data

def filt_resample(data, srate, resrate, lpfreq=256, norm='MAD'):
    '''
    resamples data to a common frequency with cubic spline interpolation
    performs 3-harmonic 60 Hz noise filtering and low-pass filtering
    

    Parameters
    ----------
    data : numpy array
        data in samples x channels.
    srate : float/int
        sampling rate.
    resrate : float/int
        resampled rate to bring everything to.
    lpfreq : float/int (optional)
        cutoff frequency for lowpass filter
        default is resrate/2, but will be set to 150 for actual analysis
    norm : str (optional)
        normalization method. defaults to 'MAD' (median absolute deviation),
        but can also be 'zscore'  
        
    Returns
    -------
    dsts : array (1-D)
        timestamps for new sampling rate
    fdata : ndarray
        filtered and standardized dataset
    filt_win : ndarray
        boolean mask that is true for values within range, false for values
        that are putative artifact
    '''
    if norm is None:
        pass
    else:
        assert norm in ['MAD','zscore'],"check norm; is it 'MAD' or 'zscore'?"
    
        
    # Resample and filter
    recdur = data.shape[0] / srate
    ts = np.linspace(1/srate, recdur, data.shape[0])
    
    # 2^n interpolation scheme and downsampling    
    upsrate = 2**np.ceil(np.log2(srate))
    if upsrate <= resrate:
        upsrate = resrate * 2
    
    # new timing vector    
    xq = np.arange(1/upsrate, ts[-1], 1/upsrate)
    # interpolate - CHECK TRANSPOSING, MAY BE WEIRD HOLDOVER FROM MATLAB
    itpdata = np.zeros((len(xq),data.shape[1]))
    # interpolate (is looping really necessary or is there a better fxn?) 
    for i in range(data.shape[1]):
        itpdata[:,i] = splev(xq, splrep(ts, data.to_numpy()[:,i]))
    # Lowpass itpdata at resrate/2 to avoid aliasing
    itpdata = bfilt(itpdata, upsrate, 4, resrate/2, "low")
    ds = int(upsrate / resrate)
    # downsampled timestamps
    dsts = xq[::ds]
    
    # nested filters:
    # 1) 180 Hz bandstop (3rd 60 Hz harmonic)
    # 2) 150 Hz bandstop (2nd 60 Hz harmonic)
    # 3) 150 Hz lowpass
    # 4) 60 Hz bandstop (line-noise)
    datafilt = bfilt(bfilt(bfilt(bfilt(itpdata[::ds,:],resrate,6,[179,181],'bandstop'),
                     resrate,6,[149,151],'bandstop'),resrate,6,lpfreq,'lowpass'),
                     resrate,4,[59,61],'bandstop')
    
    # Identify artifact
    filt_wins = np.zeros(datafilt.shape,dtype=bool)
    for i in range(data.shape[1]):
        high_amps = np.abs((datafilt[:,i]-np.median(datafilt[:,i]))/MAD(datafilt[:,i]))>10
        amp_wins = rolling_sum(high_amps,int(resrate/8)) # 0.125 seconds
    
        # create a mask that identifies high amplitude windows
        filt_wins[:,i] = np.logical_not(amp_wins>16)

    # normalize "manually" to mask influence of artifacts
    if norm == 'MAD':
        med = np.median(datafilt[filt_wins],axis=0)
        medAD = MAD(datafilt[filt_wins],axis=0)
        normdata = (datafilt-med)/medAD
    elif norm == 'zscore':
        avg = np.mean(datafilt[filt_wins],axis=0)
        std = np.std(datafilt[filt_wins],axis=0)
        normdata = (datafilt-avg)/std
    else:
        print("not normalizing")
        normdata = datafilt

    # apply small Gaussian filter to nearby points
    # really doesn't do much
    # normdata = gfilt(normdata,1,axis=0,order=0,mode='nearest',radius=2)
    return dsts, normdata, filt_wins

def calc_spectra(data, srate, t_res=1/8, f_res=0.5):
    '''
    calculate spectra spaced by t_res seconds at f_res using welch's method
    NOTE - welch params are hardcoded for consistency! change them here
           (not going to add them as kwargs)
    
    Parameters
    ----------
    data : numpy array (should be vector!)
        data in samples x channels. (just one channel!)
        NOTE - expecting one bipolar re-referenced channel
    srate : float/int
        sampling rate.
        NOTE - make sure it matches array, especially if resampled!!
    t_res : float
        time between spectra centers in seconds
        default is 1/8
    f_res : float
        resolution for frequency spacing in Hz
        NOTE - set by length of window/number of samples
        default is 0.5 (2 second window; nperseg set to srate)
        (maybe switch this to timewin in seconds?)

    Returns
    -------
    spectra : ndarray (len(centers) x len(f))
        time frequency matrix of power spectra
    f : vector (srate/2 * 1/f_res)
        frequency bins
    centers : ~len(data * t_res)
        time point index each spectral calculation is centered around
        NOTE - np.unique(np.diff(centers))/srate == t_res
    ''' 
    
    # get ixs to calculate ffts (via welch) for data segments
    dtsamps = srate*t_res # time resolution, in samples
    # number of samples per fft window
    nsampswin = int(srate/f_res)
    # center points (in samples - drop the first and last seconds of data)
    centers = np.arange(nsampswin/2,data.shape[0]-nsampswin/2,dtsamps,dtype=int)
    allsamps = np.zeros((len(centers),nsampswin),dtype=int)
    for ix,c in enumerate(centers):
        # grab 1/2 windows around center point for each dt
        allsamps[ix,:] = np.arange(c-(nsampswin/2),c+(nsampswin/2),dtype=int)
    
    # reshape the data
    redata = data[allsamps]
    
    # calculate PSD for centers
    f,spectra = welch(redata, fs=srate, window='hann', nperseg=nsampswin, 
                      noverlap=nsampswin*0.875, nfft=nsampswin, detrend=False)
    return spectra,f,centers

def bipolar_reref(data,srate,resrate,lpfreq=100, norm="zscore"):
    '''
    Parameters
    ----------
    data : 2-d array (n x 2)
        channel pairs for re-referencing
    srate : float/int
        sampling rate (original)
    resrate : float/int
        resampled rate to bring everything to.
    lpfreq : float/int (optional)
        cutoff frequency for lowpass filter
        default is resrate/2, but will be set to 150 for actual analysis
    norm : string
        can be either "zscore" or "MAD"
        defaults to "zscore"

    Returns
    -------
    1 dimensional np array of bipolar re-referenced data.
    ts = resampled timestamps (starting at 0)
    ## to-do: most data
    '''
    
    # remove baseline drift with 1 second rolling average
    hpfdata = data-data.rolling(window=int(srate),min_periods=int(srate/2),center=True).mean()
    # filter 60 Hz line noise, harmonics, and lowpass. then resample and normalize traces
    [ts,normdata,filt_wins] = filt_resample(hpfdata, srate, resrate, lpfreq=lpfreq, norm=norm)
    # bipolar re-referencing pair from region
    return(ts,np.squeeze(np.diff(normdata,axis=1)))