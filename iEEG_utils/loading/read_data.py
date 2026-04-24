# -*- coding: utf-8 -*-
"""
Created on Thu May  1, 2025
edits feb 2026
major edits apr 24, 2026

@author: JTM
"""

import pandas as pd
import numpy as np
import os
import re
import warnings
import tkinter as tk
from tkinter import filedialog

def select_directory(title="Select Folder"):
    """
    Opens a dialog box to select a directory and returns the absolute path.
    should define what is later referred to as "fstr"
    """
    # Initialize tkinter and hide the main root window
    root = tk.Tk()
    root.withdraw()
    
    # Ensure the dialog appears on top of other windows
    root.attributes('-topmost', True)
    
    # Open the directory selection dialog
    # Returns the absolute path as a normal string
    selected_path = filedialog.askdirectory(title=title)
    
    # Destroy the root window to clean up resources
    root.destroy()
    
    # Normalize the path for the current OS (e.g., handles backslashes on Windows)
    # This ensures the path behaves like a 'raw' string for filesystem operations
    return os.path.normpath(selected_path) if selected_path else ""

def load_iEEG(fstr, load_meta=True, chs=None):
    """
    Load iEEG data and optionally sampling-rate metadata.

    The function always returns `(srate, data)`. If metadata is requested
    but not found (in `fstr` or its parent directory), a warning is issued
    and `srate` is returned as None. iEEG data loading is mandatory.

    Parameters
    ----------
    fstr : str
        Path to directory containing an '*iEEG.csv' file.
    load_meta : bool, default True
        Whether to attempt loading metadata (sampling rate and montage).
    chs : list of str or None
        Channels to load from the CSV file. If None and metadata is
        available, channels are inferred from the montage.

    Returns
    -------
    srate : float or None
        Sampling rate in Hz if metadata is found, otherwise None.
    data : pandas.DataFrame
        iEEG time-series data.

    Raises
    ------
    FileNotFoundError
        If no iEEG CSV file is found in `fstr`.
    ValueError
        If metadata is found but does not contain exactly one sampling rate.
    """

    srate = None

    # look for metadata (channels file) in supplied folder, then parent
    if load_meta:
        try:
            # supplied directory (fstr)
            chtab = load_info(fstr, ftype="channels")
        
        except FileNotFoundError:
            # not there either, assuming it doesn't exist
            warnings.warn("Channels metadata not found; proceeding without metadata.",
                        RuntimeWarning)
            chtab = None

        if chtab is not None:
            freqs = chtab.sampling_frequency.dropna().unique()
            if len(freqs) != 1:
                raise ValueError(
                    "Expected a single unique sampling frequency "
                    "in channels metadata."
                )

            srate = freqs[0]
            montage = load_info(fstr, ftype="montage")
            chs = montage.name.tolist() + ["time"]

    # load iEEG data (required)
    for fname in os.listdir(fstr):
        if re.search(r"iEEG\.csv$", fname, re.IGNORECASE):
            path = os.path.join(fstr, fname)
            if chs == None:
                data = pd.read_csv(path, index_col=0)
            else:
                data = pd.read_csv(path, index_col=0,usecols=chs)
            return srate, data

    raise FileNotFoundError(f"No iEEG.csv found in directory: {fstr}")

def load_info(fstr, ftype="montage"):
    """
    Standalone that just loads metadata
    Useful for checking on basic data descriptions before loading

    Parameters
    ----------
    fstr : string
        string pointing to the FOLDER with meatadata file to load.
        Looks for '*channels.csv', "*events.csv", or '*montage.csv' file with basic metadata
        NOTE - both files are comma delimited (hence, csv)
    ftype (kwarg) : string
        defaults to "montage", but can also be "events" or "channels"
        (honestly, not a strict check here, just looking for fstr that ends with csv
         so, go wild with it and expect errors)
        

    Returns
    -------
    pandas dataframe (returned directly, no variable created first)
        full table containing metatdata
        NOTE - this behavior is different from load_iEEG, which just returns
               the sampling rate from the table!
    """
    try:
        for f in os.listdir(fstr):
            tab = re.search(fr"{ftype}\.csv$", f, re.IGNORECASE)
            if tab:
                return pd.read_csv(os.path.join(fstr, f), delimiter=",")
        if not tab:
            raise FileNotFoundError("Couldn't find "+ftype+" file...")
    
    except FileNotFoundError:
        # didn't work, try parent
        parent = os.path.dirname(fstr)
        fnames = np.array(os.listdir(os.path.dirname(fstr)))
        matches = [True if re.search(fr"{ftype}\.csv$",f,re.IGNORECASE) else False for f in fnames]
        try:
            f = fnames[matches].item()
            return pd.read_csv(os.path.join(parent,f), delimiter=",")
        except:
            raise FileNotFoundError("Couldn't find "+ftype+" file...")
        
