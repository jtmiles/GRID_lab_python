# -*- coding: utf-8 -*-
"""
Created on Thu May  1, 2025
edits feb 2026

@author: JTM
"""

import pandas as pd
import os
import re
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
    Load iEEG (and meta) data from .csv file, given a folder location

    Parameters
    ----------
        fstr : string
           string pointing to the FOLDER with iEEG data file to load.
           The iEEG file should be .csv 
        load_meta : bool
            True will look for corresponding metadata table in 'channels.txt'.
            Default is True because this is where sampling rate is stored.
            NOTE - channels file is tab delimited
        chs : list of strings
            name of channels to load
            NOTE - must match column names in  iEEG.csv

    Returns:
        srate : float
            samples per second. expecting common sampling rate across channels
        data : ndarray
            raw data saved in .csv file.    
    """
    for f in os.listdir(fstr):
        iEEG = re.search(r"iEEG.csv",f,re.IGNORECASE)
        if iEEG:
            if chs is None:
                data = pd.read_csv(fstr+"\\"+f)
            else:
                data = pd.read_csv(fstr+"\\"+f,usecols=chs)
        if load_meta:
            chtab = load_info(fstr)
            assert len(chtab.sampling_frequency.unique()) == 1, "expecting a single sampling frequency"
            srate = chtab.sampling_frequency.unique()[0]
           
    if load_meta:
        return srate, data
    else:
        return data

def load_info(fstr):
    """
    Standalone that just loads metadata
    Useful for checking on basic data descriptions before loading

    Parameters
    ----------
    fstr : string
        string pointing to the FOLDER with meatadata file to load.
        Looks for '*channels.csv' file with basic metadata annotations
        NOTE - channels file is comma delimited (hence, csv)

    Returns
    -------
    pandas dataframe (returned directly, no variable created first)
        full table containing metatdata
        NOTE - this behavior is different from load_iEEG, which just returns
               the sampling rate from the table!
    """
    for f in os.listdir(fstr):
        channels = re.search(r"channels.csv",f)
        if channels:
            return pd.read_csv(fstr+"\\"+f,delimiter=",")
        
def load_montage(fstr):
    """
    Standalone that loads pre-made montage info
    Useful for adding context to iEEG data (sbj age, regions, ch names etc.)

    Parameters
    ----------
    fstr : string
        string pointing to the FOLDER with meatadata file to load.
        Looks for '*channels.csv' file with basic metadata annotations
        NOTE - channels file is comma delimited (hence, csv)

    Returns
    -------
    pandas dataframe (returned directly, no variable created first)
        table with channel metatadata (| ID | age | region | ch | name |)
        NOTE - this behavior is different from load_iEEG, which just returns
               the sampling rate from the table!
    """
    for f in os.listdir(fstr):
        montage = re.search(r"montage.csv",f)
        if montage:
            return pd.read_csv(fstr+"\\"+f,delimiter=",")
        