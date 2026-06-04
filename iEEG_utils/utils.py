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


