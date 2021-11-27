"""
Function to fully analyse rems for each eye & direction of movement
"""
import os
from pathlib import Path
import mne
import numpy as np
import pandas
from matplotlib import pyplot as plt
import scipy
import neurokit2 as nk
from rems_functions import *
import warnings
warnings.filterwarnings("ignore")

def rems_analyse(signal, peaks_troughs, channel_crossings, eye, direction, e1_peaks, e1_troughs, e2_peaks, e2_troughs, invert=False):
    rems_start_crossings, rems_end_crossings, start_to_peak_distances, peak_to_end_distances = \
        match_crossing_to_peaks(peaks_troughs, channel_crossings)

    epochs = epoch_data(signal, peaks_troughs, sampling_freq, invert)  # from this function on, trough signals are inverted to peaks

    count, loc_min_peak_diff, rems_local_minima = \
        find_landmarks(epochs, peaks_troughs, rems_start_crossings, rems_end_crossings, start_to_peak_distances,
                       peak_to_end_distances, sampling_freq)

    rems_df = rems_characteristics(final_detected_p_t, episode, peaks, rems_start_crossings, rems_end_crossings,
                            rems_local_minima, e1_peaks, e2_peaks, e1_troughs, e2_troughs, eye, direction)
    return rems_df