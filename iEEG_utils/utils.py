# -*- coding: utf-8 -*-
"""
Created on apr 24, 2026

Updates:
  May 05, 2026

@author: JTM
"""

import re
import ipywidgets as widgets
import numpy as np

from scipy.signal import hilbert
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation as MAD

from iEEG_utils.processing.filtering import rolling_sum

def find_valid_pairs(ch_list):
  '''
  Looks through a list of channel names (from a montage file) and identifies
  which are next to one another on an electrode (same electrode name and one
  number apart for the contact number, e.g., LA 1 and LA 2)

  ch_list is a list of channel name strings
    > USE A LIST SO APPEND WORKS
  entries should be formated as:
    > 2 or 3 letters
    > either a space, typical separator("_" and "-" are most common), or no space
    > a number

  returns a list of valid contact pairs (string names of the contacts)
  '''
  pattern = re.compile(r'^([A-Za-z]{1,3})[\s_-]*(\d+)$')
  result = []

  # Work from the end toward the front
  while len(ch_list) >= 2:
      current = ch_list.pop()      # always remove the last element
      previous = ch_list[-1]       # peek at the new last element

      m1 = pattern.match(current)
      m2 = pattern.match(previous)

      if not m1 or not m2:
          continue

      prefix1, num1 = m1.group(1), int(m1.group(2))
      prefix2, num2 = m2.group(1), int(m2.group(2))

      if prefix1 == prefix2 and abs(num1 - num2) == 1:
          result.append((previous, current))

  return result

def dropdown_select( names, label="Choose region:",default=None):
  """
  Create and display an interactive dropdown widget in a Jupyter notebook.

  Parameters
  ----------
  names : iterable of str
      The items to show in the dropdown.
  label : str, optional
      Label shown next to the dropdown.
  default : str or None, optional
      Default selected value (must be in names).

  Returns
  -------
  ipywidgets.Dropdown
      The created dropdown widget.
  """
  names = list(names)

  if default is not None and default not in names:
      raise ValueError("default must be one of the provided names")

  dropdown = widgets.Dropdown(
      options=names,
      value=default if default is not None else names[0],
      description=label,
      disabled=False,
  )

  display(dropdown)
  return dropdown

def phase_scramble(x, seed=None):
  """
  Phase-scramble data along specified FFT axes.

  Parameters
  ----------
  x : ndarray
      Input array (any dimensionality, but expecting the final dimensions to contain segments)
  seed : int or None
      RNG seed for reproducibility.

  Returns
  -------
  x_scrambled : ndarray
      Phase-scrambled (real) array with same shape as x.
  """
  rng = np.random.default_rng(seed)

  # FFT along selected axis
  fft_x = np.fft.rfft(x, axis=-1)

  # Random phase with same shape in Fourier space
  random_phase = rng.uniform(-np.pi, np.pi, size=fft_x.shape)
  # do NOT randomize the DC component
  random_phase[...,0] = 0.0
  # (or nyquist frequency, if it's there)
  if x.shape[-1] % 2 == 0:
          random_phase[..., -1] = 0.0

  # Randomize Fourier phases while preserving magnitudes
  fft_scrambled = fft_x * np.exp(1j * random_phase)

  # Inverse FFT, return real signal
  return np.fft.irfft(fft_scrambled, axis=-1, n=x.shape[-1])

def envelope_noise_screen(signal, srate, window_s=1.0):
  """
  Envelope-based noise detector using robust statistics.

  Parameters
  ----------
  signal : 1D ndarray
      Pre-bandpassed signal (single channel).
  srate : float
      Sampling rate (Hz).
  window_s : float
      Window (seconds) for rolling envelope median.
  z_thresh : float
      Robust z-score threshold.
  min_duration_s : float
      Minimum duration (seconds) above threshold.

  Returns
  -------

  noise_mask (*deprecated*): 1D boolean ndarray
      True where signal is flagged as noisy.
  z_env : 1D ndarray
      Robust z-scored envelope statistic.
  """

  win_samples = int(window_s * srate)
  if win_samples < 1:
      raise ValueError("window_s too small")
  # get signal RMS  
  rms = np.sqrt(rolling_sum(signal**2,win_samples)/win_samples)  

  # take its rolling median
  rms_med = median_filter(rms, size=win_samples, mode="nearest")
  # calculate the baseline (median) of RMS values
  baseline = np.median(rms)
  # scale by the median absolute deviation (MAD)
  scale = MAD(rms)
  if scale == 0:
      scale = np.finfo(float).eps
  z_env = (rms_med - baseline) / scale


  return z_env

def spike_surrogates(signal, ix_mat, n_samps, win, seed=None):
  """
  Estimating spike thresholds using line-length calculations across
  surrogate data segments.

  Parameters
  ----------
  signal : 1D ndarray
      Pre-bandpassed signal (single channel).
  ix_mat : 2D array
      Array of timestamp indices that will be used to reshape signal.
      (N segments x M samples per segment - note: N is often trials)
  n_samps : int
      Number of times to randomly sample (rows) from ix_mat.
      Sets the number of entries into the null distribution.
  win : int
      number of samples to take for rolling sum in line-length calculation

  Returns
  -------

  null_dist : 1D array
      Distribution of surrogate segment line-length maxima
  """

  rng = np.random.default_rng(seed)

  # create new matrix of data segments for calculating null distribution
  # IMPORTANT: expecting the *0-th* dimension of ix_mat to be N segments 
  randixs = rng.integers(low=0, high=ix_mat.shape[0], size=[n_samps,])

  # calculate the surrogate samples (n_samps x M samples per segment)
  surrogates = phase_scramble(signal[ix_mat[randixs,:]])
  
  # calculate null line-lengths (across each surrogate segment) 
  null_spikes = rolling_sum(np.abs(np.diff(surrogates,axis=-1)),win,axis=-1)
  
  # return the *max* of each individual line-length calculation from each segment
  return np.max(null_spikes,axis=-1)


