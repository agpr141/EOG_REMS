"""
Simple script for loading in EOG data & playing around with peak detections and plotting

3 key things to play around with:
- peak detection parameters (lines 111-118 inclusive) : see scipy.find_peaks for more info
- peak agreement value- the difference in movement initiation between the eyes (lines 141-142, 150-151)
- data visualisation (lines 161-169 inclusive) : see matplotlib.pyplot for more info

author: @agpr141
last update: 23/11/21
"""

# import modules - pip install these!
import os
from pathlib import Path
import mne
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy
import neurokit2 as nk


# import functions
def find_nearest(array, value):
    array = np.asarray(array)
    near_idx = (np.abs(array - value)).argmin()
    return array[near_idx]


# import eog data, hypnogram & eeg data (for hypno alignment)
path = Path('Y:/22qEEG/E004-1-1-1')  # define path to participant folder
os.chdir(path)  # change working directory to path
hypnogram = 'scoring_outputs/E004-1-1-1_scoring_info_novid.csv'  # define path to hypnogram
EEG = 'exported_data/E004-1-1-1_sleep_EEG_PREP.edf'  # define path to EEG data (for hypnogram alignment timings)
EOG = 'exported_data/E004-1-1-1_PSG.edf'  # define path to EOG data
sampling_freq = 256  # define sampling frequency of EOG data

# load in EEG data to access info such as recording start time
EEG_sig = mne.io.read_raw_edf(EEG, exclude=['F5', 'FP1', 'FP2' 'F6', 'F10', 'F7', 'FC5', 'F3', 'F1', 'F2',
                                            'FC6', 'T9', 'FT7', 'FC3', 'FC1', 'FC2', 'FC4', 'FT8', 'T10',
                                            'T7',
                                            'C5', 'C3', 'C1', 'C2', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP1',
                                            'CP2', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6',
                                            'P8', 'AF4', 'P9', 'PO3', 'O1', 'PO4', 'P10', 'Oz', 'AF3', 'Fz',
                                            'FCz', 'Cz', 'Pz', 'POz' 'F9', 'F6', 'FP2', 'POz', 'F4', 'F8', 'C4', 'O2'],
                              preload=True)  # load in sleep hypno

# load in EOG data
EOG_sig = mne.io.read_raw_edf(EOG, exclude=['ChinR', 'ChinL', 'EKG', 'Thorax', 'Abdomen', 'NasalPressure',
                                            'Plethysmogram', 'Position', 'Sp02', 'Pulse'],
                              preload=True)  # load in EOG

# align hypnogram to EOG data
hypno = pd.read_csv(hypnogram, names=['onset', 'offset', 'description'])  # units = minutes
onset = pd.Series.tolist(hypno['onset'])
offset = pd.Series.tolist(hypno['offset'])
onset_secs = [x * 60 for x in onset]  # convert onset times from minutes to seconds
duration = []  # initialise duration variable
for index in range(len(offset)):
    dur = offset[index] - onset[index]  # duration = offset-onset
    duration.append(dur)  # create new 'real' duration variable
duration_secs = [x * 60 for x in duration]
description = pd.Series.tolist(hypno['description'])
orig_time = EEG_sig.info['meas_date']  # find EEG original time for alignment
annot = mne.Annotations(onset_secs, duration_secs, description, orig_time)  # create new annotations object
EOG_sig.set_annotations(annot, emit_warning=True)  # align annotations object with EOG
eog_annot = pd.DataFrame(EOG_sig.annotations)  # save EOG-aligned timings into new hypnogram dataframe

# filter & plot EOG signal to get a sense of how the EOG changes across sleep stages
EOG_filt = EOG_sig.copy().filter(l_freq=0.5, h_freq=10)  # filter signal for data visualisation
EOG_filt.plot(show_first_samp=True, show_scrollbars=True, show_scalebars=True, time_format='clock')

# extract all rem episodes and store data for each eye as rem_episodes_e1/e2
rem_timings = eog_annot[eog_annot['description'].str.contains('N|W|A') == False]  # find only REM episodes
epoch_buffer = 30  # buffer of 30s to introduce at the start and end of REM episode to exclude transitions
rem_episodes_e1 = []
rem_episodes_e2 = []

# extract all rem episodes
for idx, row in rem_timings.iterrows():
    eog1 = EOG_sig.copy()
    eog1.pick(picks='E1')
    eog2 = EOG_sig.copy()
    eog2.pick(picks='E2')
    rem_onset = row['onset']
    rem_offset = row['duration'] + row['onset']
    e1_segment = eog1.crop(tmin=rem_onset + epoch_buffer, tmax=rem_offset - epoch_buffer)
    e2_segment = eog2.crop(tmin=rem_onset + epoch_buffer, tmax=rem_offset - epoch_buffer)
    rem_episodes_e1.append(e1_segment._data)
    rem_episodes_e2.append(e2_segment._data)

# peak detect
# select which episode you want to peak detect on
e1_ep = rem_episodes_e1[5]  # change this '0' to whatever episode you want to look at
e1_ep = np.ndarray.flatten(e1_ep)
e1_z = scipy.stats.zscore(e1_ep)
e2_ep = rem_episodes_e2[5]  # change this '0' to whatever episode you want to look at
e2_ep = np.ndarray.flatten(e2_ep)
e2_z = scipy.stats.zscore(e2_ep)

# apply cleaning algorithm to each eog channel (filters)
clean_e1 = nk.eog_clean(e1_z, sampling_rate=256, method='neurokit')
e1_std = np.std(clean_e1)  # calculate standard dev
clean_e2 = nk.eog_clean(e2_z, sampling_rate=256, method='neurokit')
e2_std = np.std(clean_e2)  # calculate standard dev

# find channel crossings where E1/E2 intersect for later detections
channel_crossings = np.argwhere(np.diff(np.sign(clean_e1 - clean_e2))).flatten()

# choose your parameters for peak detection
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
height = None  # height of peaks
threshold = None  # peak threshold (vertical distance from neighbouring samples)
distance = None  # horizontal distance between peaks (in samples)
prominence = 1.5*e1_std  # prominence of peaks (vertical distance between peak & lowest contour line)
width = None  # width of peaks in sample
wlen = None  # used w/ peak_prominences - read guidance online
rel_height = None  # used with width - read guidance online
plateau_size = None  # size of the flat top of peaks (in samples)

# E1 peak detection - to change thresholds look at scipy.find_peaks() for options
e1_peaks, e1_properties = scipy.signal.find_peaks(clean_e1, prominence=prominence)  # change these parameters
# E1 trough detection
e1_invert = clean_e1 * -1  # invert signal so troughs become peaks & can use find_peaks algorithm
e1_troughs, e1_trough_properties = scipy.signal.find_peaks(e1_invert, prominence=prominence)  # change these parameters
# E2 peak detection
e2_peaks, e2_properties = scipy.signal.find_peaks(clean_e2, prominence=prominence)  # change these parameters
# E2 trough detection
e2_invert = clean_e2 * -1  # invert signal so troughs become peaks & can use find_peaks algorithm
e2_troughs, e2_trough_properties = scipy.signal.find_peaks(e2_invert, prominence=prominence)  # change these parameters

# compare peak/trough indexes between eyes. keep those in agreement(w/i 76 (~300ms) samples of each other - Porte 2004)
matched_peaks = []
matched_troughs = []
e1_peaks_m = []
e1_troughs_m = []
e2_peaks_m = []
e2_troughs_m = []
for index in e1_peaks:
    match = find_nearest(e2_troughs, index)
    idx_plus = index + 76  # edit this number (# samples between a peak&trough)
    idx_minus = index - 76  # edit this number (# samples between a peak&trough)
    if match >= idx_minus and match <= idx_plus:
        matched_peaks.append(match)
        e1_peaks_m.append(index)
        e2_troughs_m.append(match)

for index in e1_troughs:
    match = find_nearest(e2_peaks, index)
    idx_plus = index + 76  # edit this number (# samples between a peak&trough)
    idx_minus = index - 76  # edit this number (# samples between a peak&trough)
    if match >= idx_minus and match <= idx_plus:
        matched_troughs.append(match)
        e1_troughs_m.append(index)
        e2_peaks_m.append(match)

# plot detected peaks ({eye}_{peaks/troughs}_m) on e1 / e2 channel
# uses matplotlib pyplot - details of how to change plots here https://matplotlib.org/stable/api/pyplot_summary.html
plt.figure()
plt.plot(clean_e1, linewidth=0.8, color='navy', label='EOG1')
plt.plot(clean_e2, linewidth=0.8, color='mediumblue', label='EOG2')
plt.plot(e1_peaks_m, clean_e1[e1_peaks_m], 'o', color='darkred', label='EOG1 Peaks')
plt.plot(e2_peaks_m, clean_e2[e2_peaks_m], 'o', color='darkorange', label='EOG2 Peaks')
plt.plot(e1_troughs_m, clean_e1[e1_troughs_m], 'D', color='darkorange', label='EOG1 Troughs')
plt.plot(e2_troughs_m, clean_e2[e2_troughs_m], 'D', color='darkred', label='EOG2 Troughs')
plt.title('REM episode detections')
plt.legend(loc='upper right')
