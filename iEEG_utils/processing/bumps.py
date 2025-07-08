# -*- coding: utf-8 -*-
"""
Created on Wed May 28 09:36:47 2025

@author: jmile3
"""
import numpy as np
from scipy.signal import find_peaks
import pandas as pd

# set default frequency bins (0.5 to 100 in 0.5 Hz increments)
freqs = np.linspace(0.5,100,200)

def find_bumps(resid_trace,pk_ix,distance=2,prominence=0.01):
    '''
    
    '''
    
    sigsigns = np.sign(resid_trace) # sign of signal (-1, 0, +1)
    # find where neighboring indices do not match (zero crossings)
    xingbool = np.insert(sigsigns[:-1] != sigsigns[1:],0,False)
    # get zero crossing before pk_ix
    xingixs = (np.array(np.nonzero(xingbool)))
    prexings = xingixs[xingixs<=pk_ix]
    # get local min before pk_ix
    pre_min,_ = find_peaks(-resid_trace[:pk_ix],distance=distance, prominence=prominence)
    if (np.size(prexings) > 0) & (np.size(pre_min) == 0):
        prepk = max(prexings) # just get last x-ing if multiple
        # print("pre 1")
    elif (np.size(prexings) == 0) & (np.size(pre_min) > 0):
        prepk = max(pre_min)
        # print("pre 2")
    elif (np.size(prexings) > 0) & (np.size(pre_min) > 0):
        pre_xing = max(prexings)
        pre_min = max(pre_min)
        # check to see if zero x-ing or pre_min is closer to pk_ix
        if (pk_ix-pre_xing) < (pk_ix-pre_min):
            prepk = pre_xing
            # print("pre 3")
        else:
            prepk = pre_min
            # print("pre 4")
    else:
        # print("bump does not begin in frequency range")
        prepk = np.nonzero(resid_trace[:pk_ix]==min(resid_trace[:pk_ix]))
            
    # get zero crossing after pk_ix
    postxings = xingixs[xingixs>pk_ix] # index values
    # local min after pk_ix
    post_mins,_ = find_peaks(-resid_trace[pk_ix:],distance=distance, prominence=prominence)
    if (np.size(postxings) > 0) & (np.size(post_mins) == 0):
        postpk = min(postxings) # just get first x-ing if multiple
        # print("post 1")
    elif (np.size(postxings) == 0) & (np.size(post_mins) > 0):
        postpk = min(post_mins)+pk_ix
        # print("post 2")
    elif (np.size(postxings) > 0) & (np.size(post_mins) > 0):
        post_xing = min(postxings)
        post_min = min(post_mins)+pk_ix # new variable (ensures single value)!!
        # check to see if zero x-ing or post_min is closer to pk_ix
        if (post_xing-pk_ix) < (post_min-pk_ix):
            postpk = post_xing
            # print("post 3")
        else:
            postpk = post_min
            # print("post 4")
    else:
        # print("bump does not end in frequency range")
        postpk = np.nonzero(resid_trace[pk_ix:]==min(resid_trace[pk_ix:]))+pk_ix
        
    if prepk==pk_ix:
        prepk = prepk-1
    if postpk==pk_ix:
        postpk = postpk+1
        
    # squeezing b/c find_peaks returns arrays, just want single integer values
    return [np.squeeze(prepk), np.squeeze(pk_ix), np.squeeze(postpk)]

def get_bump_ixs(lowcut,hicut,resid_trace,keep_max=True,freqs=freqs,height=0.025,
                 prominence=0.05,width=(3,40),rel_height=1.0,distance=4):
    '''
    use the zero-xing and peak finding from find_bumps to identify
    peak frequencies in a 1/f-corrected power spectrum
    
    currently only allows for maximum of 2 peaks between lowcut and hicut
    
    Parameters
    ----------
    lowcut : float
        frequency value to start search at (typically 1 or 2 Hz for alpha)
    hicut : float
        frequency value to end search at (typically 15 or 16 Hz for alpha)
    resid_trace : array (1-D)
        array (vector) of residual power after 1/f correction
    keep_max : boolean (default = True)
        True returns just the maximum peak if more than one between low/hicut
    freqs : array (1-D)
        array (vector) of frequency values 
        NOTE - default 0.5:0.5:100 set at beginning of script
    height : float
        height from the scipy.signal.find_peaks function
    prominence : float
        from the scipy.signal.find_peaks function
    width : int/float or tuple/iterable (min, max)
        from the scipy.signal.find_peaks function
    distance : int/float(?)
        from the scipy.signal.find_peaks function

    Returns
    -------
    int (if keep_max = True)
        index of max bump in spectrum (or np.nan if no peak found)
    3 ints to unpack
        lowf_bump : ix of low frequency peak
        highf_bump : ix of high frequency peak
        n_bumps : number of bumps (0, 1, or 2)
    '''
    # only searching between locut and hicut
    fwin = freqs[freqs<=hicut]
    pk_ixs,_ = find_peaks(resid_trace[(freqs>=lowcut)&(freqs<=hicut)], height=height, prominence=prominence,
                          width=width, distance=distance)
    pk_ixs = pk_ixs + np.nonzero(fwin==lowcut) # add index value of lowcut
    n_bumps = np.size(pk_ixs)
    
    lowf_bump = np.zeros(3,dtype=int) # initialize as zeros
    highf_bump = np.zeros(3,dtype=int) 
    # should be 0, 1, or 2 peaks in the theta/alpha range
    if np.size(pk_ixs) == 0:
        # print("no peaks")
        if keep_max:
            return 0
        else:
            return lowf_bump, highf_bump, n_bumps
    elif np.size(pk_ixs) >= 1:
        if np.size(pk_ixs) == 1:
            lowf_bump[:] = find_bumps(resid_trace,pk_ixs[0,0])
            if keep_max:
                # just return the pk_ix
                return lowf_bump[1]
            else:
                return lowf_bump, highf_bump, n_bumps
        elif np.size(pk_ixs) >= 2:
            # stop at second peak
            lowf_bump[:] = find_bumps(resid_trace[:pk_ixs[0,1]],pk_ixs[0,0])
            # start at the peak of the lowf bump and go to the end of the trace
            highf_bump[:] = find_bumps(resid_trace[lowf_bump[1]:],pk_ixs[0,1]-lowf_bump[1])
            highf_bump = highf_bump+lowf_bump[1] # add index of lowf peak
            if keep_max:
                # compare and return larger
                if resid_trace[lowf_bump[1]]>resid_trace[highf_bump[1]]:
                    return lowf_bump[1]
                else:
                    return highf_bump[1]
        # elif np.size(pk_ixs) > 2:
        #     print("too many peaks")
            else:
                return lowf_bump, highf_bump, n_bumps

def demo_df(spect_table, ix, resid_trace,keep_max=False, 
            freqs=freqs, cols=['ID','age','region'], lowcut=2, hicut=16,
            height=0.025,prominence=0.05,width=(3,40),distance=4):
    '''
    for each bump in trace, get:
    peak frequency (pk_freq)
    peak amplitude (pk_amp)
    integrated power (int_pow)
    
    (essentially deprecated since adding "keep_max" to get_bump_ixs)
    '''
    bump1, bump2, n_bump = get_bump_ixs(lowcut,hicut,resid_trace,freqs=freqs,keep_max=keep_max,
            height=height,prominence=prominence,width=width,distance=distance)
    dfvars = ['pk_freq','pk_amp','int_pow'] # could be fxn input?
    # initialize a proper sized df
    # going to add the 3 identification columns plus vars, values, and bumps columns
    df = pd.DataFrame(index=range(np.size(dfvars)),columns=cols+['ix','measure','value','nbump'])
    df['ix'] = ix
    df['measure'] = dfvars
    df['value'] = 0. # np.nan (?)
    df['nbump'] = 0
    for col in cols:
        df[col] = spect_table[col][ix]
    if n_bump==0: # or ~datafilter[ix]: - removing for now; revisit
        return df.reset_index(drop=True)
    elif n_bump>=1:
        # handle first bump (guaranteed if n_bump >= 1)
        df.loc[df.measure==dfvars[0], 'value'] = freqs[bump1[1]] # pk_ix frequency of first bump
        df.loc[df.measure==dfvars[1], 'value'] = resid_trace[bump1[1]] # residual amp at pk_ix of first bump
        df.loc[df.measure==dfvars[2], 'value'] = sum(resid_trace[bump1[0]:bump1[-1]])
        df['nbump'] = 1
        if n_bump>1:
            df2 = pd.DataFrame(index=range(np.size(dfvars)),columns=cols+['measure','value','nbump'])
            df2['ix'] = ix
            df2['measure'] = dfvars
            df2['value'] = 0. # np.nan (?)
            df2['nbump'] = 2
            for col in cols:
                df2[col] = spect_table[col][ix]
            # handle second bump
            df2.loc[df2.measure==dfvars[0], 'value'] = freqs[bump2[1]] # pk_ix frequency of first bump
            df2.loc[df2.measure==dfvars[1], 'value'] = resid_trace[bump2[1]] # residual amp at pk_ix of first bump
            df2.loc[df2.measure==dfvars[2], 'value'] = sum(resid_trace[bump2[0]:bump2[-1]])
            df = pd.concat([df,df2])
        return df.reset_index(drop=True)

def get_dom_pks(pow_df):
    '''
    takes in pow_df as input and adds a dominant bump column to it
    pow_df must have nbump, measure, and value columns
    measure "pk_amp" must be listed in the measure column for comparison

    the function identifies when there are two bumps above aperiodic component,
    then, where there are, compares the peak amplitude of the two to see
    which is larger. This is considered the dominant peak.

    If there is only one bump, that is automatically marked the dominant peak
    
    (essentially deprecated since adding "keep_max" to get_bump_ixs)
    '''
    
    # find instances of 2 bump spectra (stored as 'ix' in ..._pow_df)
    ixs2bump = np.unique(np.array(pow_df.ix[pow_df.nbump==2]))
    # all 2 bump instances will have a matching 1 bump set of params for the same ix
    # eliminate the 1 bump ixs that occur as the first bump in a 2 bump spectrum from the pool of 1 bump ixs
    
    # boolean array of the FIRST bump's pk_amp for all 2-bump spectra
    amps1 = (pow_df.measure=='pk_amp') & (pow_df.nbump==1) & (np.isin(pow_df.ix,ixs2bump))
    # boolean array of the SECOND bump's pk_amp for all 2-bump spectra
    amps2 = (pow_df.measure=='pk_amp') & (pow_df.nbump==2) & (np.isin(pow_df.ix,ixs2bump))
    
    # amps1 and amps2 should be the same size, but have different indices for bump1 or bump2, respectively
    # find instances where amp for bump1 is larger
    pow_df['dom_bump'] = False # new column to flag which bump is larger
    # boolean array of larger bump1s
    bigamp1 = pow_df.loc[amps1,'value'].to_numpy()>pow_df.loc[amps2,'value'].to_numpy()
    bigix1 = ixs2bump[bigamp1] # instances of larger bump1s
    bigix2 = ixs2bump[~bigamp1] # instances of larger bump2s
    pow_df.loc[np.isin(pow_df.ix.to_numpy(),bigix1)&(pow_df.nbump==1),'dom_bump'] = True
    pow_df.loc[np.isin(pow_df.ix.to_numpy(),bigix2)&(pow_df.nbump==2),'dom_bump'] = True
    
    # find ixs with ONLY 1 bump and set dom_bump as true for nbump==1 on those spectra
    ixs1bump = np.unique(np.array(pow_df.ix[pow_df.nbump==1])) # all bump1s
    # set ixs of bump1s without a corresponding bump2
    pow_df.loc[np.isin(pow_df.ix.to_numpy(),ixs1bump)&~(np.isin(pow_df.ix,ixs2bump)),'dom_bump'] = True
    return pow_df