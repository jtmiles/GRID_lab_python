# -*- coding: utf-8 -*-
"""
Created May 21, 2026

@author: JTM
"""

import numpy as np

from scipy.signal import hilbert
from scipy.signal import find_peaks
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation as MAD

from iEEG_utils.processing.filtering import rolling_sum


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

def suppress_adjacent_small_peaks(peak_indices,prominences,
                                  max_distance=64,ratio_thresh=1.5):
  """
  Suppress smaller peaks that are temporally close to a larger adjacent peak.

  This function assumes that `peak_indices` are already sorted in ascending
  sample-index order, and that `prominences` correspond one-to-one with those
  peaks. Only adjacent peaks in the sorted list are compared.

  A peak is discarded if:
      1. it is within `max_distance` samples of an adjacent peak, and
      2. its adjacent neighbor is at least `ratio_thresh` times more prominent.

  Parameters
  ----------
  peak_indices : array-like of int
      Sorted sample indices of detected peaks.

  prominences : array-like of float
      Prominence values corresponding to `peak_indices`.

  max_distance : int, default=64 (~63 ms for sampling rate of 1024)
      Maximum allowed distance, in samples, for two adjacent peaks to be
      considered temporally nearby.

  ratio_thresh : float, default=1.5
      Minimum prominence ratio required to suppress the smaller peak.
      For example, `ratio_thresh=2.0` means one peak must be at least twice
      as prominent as its adjacent neighbor.

  Returns
  -------
  keep_mask : ndarray of bool
      Boolean mask indicating which peaks to keep.
      Use as `peak_indices[keep_mask]` and `prominences[keep_mask]`.

  """

  peak_indices = np.asarray(peak_indices)
  prominences = np.asarray(prominences)
  
  # Only compare adjacent peaks that are close enough in time.
  nearby = np.diff(peak_indices) <= max_distance

  # Signed log-ratio of adjacent prominences.
  # Positive means right peak is larger; negative means left peak is larger.
  log_ratio = np.log(prominences[1:] / prominences[:-1])

  # Require a large enough prominence difference before suppressing anything.
  different_enough = np.abs(log_ratio) >= np.log(ratio_thresh)

  # If the right peak is much larger, discard the left peak.
  discard_left = nearby & different_enough & (log_ratio > 0)
  # If the left peak is much larger, discard the right peak.
  discard_right = nearby & different_enough & (log_ratio < 0)

  discard_mask = np.zeros(len(prominences), dtype=bool)

  # Pairwise decisions map back onto the original peak positions.
  discard_mask[:-1] |= discard_left
  discard_mask[1:]  |= discard_right
  
  return ~discard_mask
      
def find_spike_heights(data,ll_mask,win,srate,
                       prominence_wlen=None,
                       min_prominence=None,
                       suppress_ratio=1,
                       suppress_dist=64):
  """
  Find positive and negative peaks in `data`, match them to an existing
  line-length threshold mask, and return matched peak indices and prominences.

  Parameters
  ----------
  data : np.ndarray
      1D signal, e.g. bipolar filtered channel.
  ll_mask : np.ndarray
      Boolean mask where line-length exceeds threshold.
      Must be same length as data.
  win : int
      Window size used to set find_peaks distance (in samples).
  srate : float
      Sampling rate in Hz.
  prominence_wlen : int or None
      Window length used by scipy for prominence calculation.
      If None, defaults to srate/4 (250 ms).
  min_prominence : float or None
      Optional prominence threshold in signal units.
      If None, prominence is calculated but not used for filtering.

  see "suppress_adjacent_small_peaks" for details on suppress_ratio (ratio_thresh)
  and suppress_dist (max_distance)

  Returns
  -------
  peak_ix : np.ndarray
      Matched peak/trough indices.
  prominence : np.ndarray
      Prominence values for matched peaks.
  """

  data = np.asarray(data, dtype=float)
  ll_mask = np.asarray(ll_mask, dtype=bool)

  if data.shape[0] != ll_mask.shape[0]:
      raise ValueError("data and ll_mask must have the same length")

  if prominence_wlen is None:
      # max window to use for the prominence calculation
      prominence_wlen = srate/4

  prom_arg = (None, None) if min_prominence is None else min_prominence

  # currently, the plan is to leave distance, width, and rel_height hardcoded
  peak_kwargs = dict(distance=int(win / 2),
                     width=(srate / 64, srate / 4),
                     rel_height=1.0,
                     prominence=prom_arg,
                     wlen=prominence_wlen)

  # Positive peaks
  pos_ix, pos_props = find_peaks(data, **peak_kwargs)
  # Negative peaks, treated as peaks in the inverted signal
  neg_ix, neg_props = find_peaks(-data, **peak_kwargs)
  peak_ix = np.concatenate([pos_ix, neg_ix])
  # return the prominences as a way to measure spike heights
  prominence = np.concatenate([pos_props["prominences"],
                               neg_props["prominences"],])
  # added May 29th for testing
  # not sure what to do with it yet, just want to remember it exists
  bases = np.unique(np.concatenate([pos_props["left_bases"],
                                    pos_props["right_bases"],
                                    neg_props["left_bases"],
                                    neg_props["right_bases"]]))
  

  # Sort by time (ix)
  order = np.argsort(peak_ix)
  peak_ix = peak_ix[order]
  prominence = prominence[order]

  # Match peaks to suprathreshold line-length mask
  ll_keep = ll_mask[peak_ix]

  # compare neighboring peak prominences to reject small peaks that
  # are part of a larger spike 
  prom_keep = suppress_adjacent_small_peaks(peak_ix, prominence,
                                            max_distance=suppress_dist,
                                            ratio_thresh=suppress_ratio)

  return peak_ix[ll_keep&prom_keep], prominence[ll_keep&prom_keep]
