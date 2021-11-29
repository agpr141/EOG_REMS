"""
script containing all functions required for REMs characterisation
author: @agpr141
last updated: 29/11/21

"""
import mne
import numpy as np
import pandas as pd
import scipy
from scipy import stats, signal
import neurokit2 as nk
from matplotlib import pyplot as plt
import statistics
import matplotlib.widgets as mwidgets
from matplotlib.widgets import Button


def extract_rem_episodes(EEG, EOG, hypnogram):
    # load in EEG data to access info such as recording start time
    EEG_sig = mne.io.read_raw_edf(EEG, exclude=['F5', 'FP1', 'FP2' 'F6', 'F10', 'F7', 'FC5', 'F3', 'F1', 'F2',
                                                'FC6', 'T9', 'FT7', 'FC3', 'FC1', 'FC2', 'FC4', 'FT8', 'T10',
                                                'T7',
                                                'C5', 'C3', 'C1', 'C2', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP1',
                                                'CP2', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6',
                                                'P8', 'AF4', 'P9', 'PO3', 'O1', 'PO4', 'P10', 'Oz', 'AF3', 'Fz',
                                                'FCz', 'Cz', 'Pz', 'POz' 'F9', 'F6', 'FP2', 'POz', 'F4', 'F8', 'C4',
                                                'O2'],
                                  preload=True)  # load in sleep hypno

    # load in EOG data
    EOG_sig = mne.io.read_raw_edf(EOG, exclude=['ChinR', 'ChinL', 'EKG', 'Thorax', 'Abdomen', 'NasalPressure',
                                                'Plethysmogram', 'Position', 'Sp02', 'Pulse'],
                                  preload=True)  # load in EOG

    # wrangle hypnogram data & match to EOG signal
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

    # segment out REM periods only (must be over 60s to be analysed), store in 'rem_periods' eeg data of ndarray
    rem_timings = eog_annot[eog_annot['description'].str.contains('N|W|A') == False]
    rem_timings.drop(rem_timings[rem_timings['duration'] <= 60]. index, inplace=True)
    epoch_buffer = 30  # buffer of 30s to introduce at the start and end of REM episode to exclude transitions
    rem_episodes_e1 = []
    rem_episodes_e2 = []
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
    return rem_episodes_e1, rem_episodes_e2, rem_timings


def check_episode_quality(rem_episodes_e1, rem_episodes_e2):
    for episode in range(len(rem_episodes_e1)):
        ax.plot(rem_episodes_e1[episode], linewidth=0.7, color='steelblue', label='EOG1')
        ax.plot(rem_episodes_e2[episode], linewidth=0.7, color='darkslateblue', label='EOG2')
        plt.legend(loc='upper right')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def manual_peak_detect(episode, rem_episodes_e1, rem_episodes_e2, bad_episode_list, good_episode_list, episode_count):
    # select nth rem episode, flatten the ndarrays for each eog channel and z-score
    e1_ep = rem_episodes_e1[episode]
    e1_ep = np.ndarray.flatten(e1_ep)  # keep units as uV to aid peak identification
    e1_z = scipy.stats.zscore(e1_ep)
    e2_ep = rem_episodes_e2[episode]
    e2_ep = np.ndarray.flatten(e2_ep)  # keep units as uV to aid peak identification
    e2_z = scipy.stats.zscore(e2_ep)
    clean_e1 = nk.eog_clean(e1_z, sampling_rate=256, method='neurokit')
    clean_e2 = nk.eog_clean(e2_z, sampling_rate=256, method='neurokit')
    e1_std = np.std(clean_e1)
    e2_std = np.std(clean_e2)
    # apply cleaning algorithm to each eog channel (filters)
    clean_e1uv = nk.eog_clean(e1_ep, sampling_rate=256, method='neurokit')
    e1_uv = clean_e1uv * 1000000
    e1_uvd = scipy.signal.detrend(e1_uv)
    clean_e2uv = nk.eog_clean(e2_ep, sampling_rate=256, method='neurokit')
    e2_uv = clean_e2uv * 1000000
    e2_uvd = scipy.signal.detrend(e2_uv)

    # find channel crossings
    channel_crossings = np.argwhere(np.diff(np.sign(clean_e1 - clean_e2))).flatten()

    # highlight peaks OR mark bad episode
    def on_click(event):
        ax.set(facecolor='indianred')
        bad_episode_list.append(episode_count)

    def onselect(vmin, vmax):
        peak_range_start.append(vmin)
        peak_range_end.append(vmax)

    peak_range_start = []  # only update with values when another bit of code is run for some reason
    peak_range_end = []  # only update with values when another bit of code is run for some reason
    fig, ax = plt.subplots()
    ax.plot(e1_uvd, linewidth=0.8, color='navy', label='EOG1')
    ax.plot(e2_uvd, linewidth=0.8, color='mediumblue', label='EOG2')
    plt.axhline(y=50, color='black')
    plt.axhline(y=-50, color='black')
    plt.title(label=['Mark *PEAKS* (drag cursor): Episode', episode])
    plt.xlabel('Samples', fontsize=10)
    plt.ylabel('Voltage (uV)', fontsize=10)
    ax.legend(loc='upper right')
    ax.set(facecolor='whitesmoke')
    plt.ylim((-300, 300))
    axes = plt.axes([0.7, 0.9, 0.2, 0.075])
    bbad = Button(axes, 'Bad Episode', color="indianred")
    bbad.on_clicked(on_click)

    rectprops = dict(facecolor='mediumaquamarine', alpha=0.5)
    span = mwidgets.SpanSelector(ax, onselect, 'horizontal',
                                 rectprops=rectprops, span_stays=True)
    plt.show(block=True)

    # find local maximum index (closest peak to start point) & append to peak_indexes list
    e1_peak_indexes = []
    e1_trough_indexes = []
    e2_peak_indexes = []
    e2_trough_indexes = []
    for index in range(len(peak_range_start)):
        start = int(peak_range_start[index])
        end = int(peak_range_end[index])
        true_start = min(start, end)
        true_end = max(start, end)
        loc_max_idx, peaks_intbng = find_peak_manual(e1_uvd, true_start, true_end)
        if peaks_intbng:
            e1_peak_indexes.append(loc_max_idx)
        elif not peaks_intbng:
            e1_trough_indexes.append(loc_max_idx)
        loc_max_idx, peaks_intbng = find_peak_manual(e2_uvd, true_start, true_end)
        if peaks_intbng:
            e2_peak_indexes.append(loc_max_idx)
        elif not peaks_intbng:
            e2_trough_indexes.append(loc_max_idx)

    if len(peak_range_start) > 0:
        good_episode_list.append(episode_count)
    return peak_range_start, peak_range_end, clean_e1, clean_e2, e1_peak_indexes, e1_trough_indexes, e2_peak_indexes, e2_trough_indexes, channel_crossings, bad_episode_list, good_episode_list, e1_uvd, e2_uvd


def find_peak_manual(sig, start, end):
    if sig[start:end].mean() > 0:  # if mean of signal slice is positive (greater than 0) - peaks
        maximums = scipy.signal.argrelextrema(sig[start:end], np.greater)
        if maximums[0].size == 0:
            loc_max_idx = end
        else:
            loc_max_idx = start + maximums[0][0]
        peaks_intbng = True  # peaks?! (interrobang) set to True as it was peaks detected
    elif sig[start:end].mean() < 0:  # if mean of signal slice is negative (less than 0) - troughs
        segment = sig[start:end]
        slice_pos = segment * -1
        maximums = scipy.signal.argrelextrema(slice_pos, np.greater)
        if maximums[0].size == 0:
            loc_max_idx = end
        else:
            loc_max_idx = start + maximums[0][0]
        peaks_intbng = False  # peaks?! (interrobang) set to False as it was troughs detected
    return loc_max_idx, peaks_intbng


def matched_peaks_detection(episode, rem_episodes_e1, rem_episodes_e2):
    # select nth rem episode, flatten the ndarrays for each eog channel and z-score
    e1_ep = rem_episodes_e1[episode]
    e1_ep = np.ndarray.flatten(e1_ep)
    e1_z = scipy.stats.zscore(e1_ep)
    e2_ep = rem_episodes_e2[episode]
    e2_ep = np.ndarray.flatten(e2_ep)
    e2_z = scipy.stats.zscore(e2_ep)

    # apply cleaning algorithm to each eog channel (filters)
    clean_e1 = nk.eog_clean(e1_z, sampling_rate=256, method='neurokit')
    e1_std = np.std(clean_e1)
    clean_e2 = nk.eog_clean(e2_z, sampling_rate=256, method='neurokit')
    e2_std = np.std(clean_e2)
    # find channel crossings where E1/E2 intersect for later detections
    channel_crossings = np.argwhere(np.diff(np.sign(clean_e1 - clean_e2))).flatten()

    # E1 peak detection - to change thresholds look at scipy.find_peaks()
    e1_peaks, e1_properties = scipy.signal.find_peaks(clean_e1, prominence=(1.5 * e1_std))
    # E1 trough detection
    e1_invert = clean_e1 * -1  # invert signal so troughs become peaks & can use find_peaks algorithm
    e1_troughs, e1_trough_properties = scipy.signal.find_peaks(e1_invert, prominence=(1.5 * e1_std))
    # E2 peak detection
    e2_peaks, e2_properties = scipy.signal.find_peaks(clean_e2, prominence=(1.5 * e2_std))
    # E2 trough detection
    e2_invert = clean_e2 * -1  # invert signal so troughs become peaks & can use find_peaks algorithm
    e2_troughs, e2_trough_properties = scipy.signal.find_peaks(e2_invert, prominence=(1.5 * e2_std))

    # compare peak/trough indexes between eyes. keep those in agreement(w/i 76 (~330ms) samples of each other)
    matched_peaks = []
    matched_troughs = []
    e1_peaks_m = []
    e1_troughs_m = []
    e2_peaks_m = []
    e2_troughs_m = []
    for index in e1_peaks:
        match = find_nearest(e2_troughs, index)
        idx_plus = index + 76
        idx_minus = index - 76
        if match >= idx_minus and match <= idx_plus:
            matched_peaks.append(match)
            e1_peaks_m.append(index)
            e2_troughs_m.append(match)

    for index in e1_troughs:
        match = find_nearest(e2_peaks, index)
        idx_plus = index + 76
        idx_minus = index - 76
        if match >= idx_minus and match <= idx_plus:
            matched_troughs.append(match)
            e1_troughs_m.append(index)
            e2_peaks_m.append(match)
    return clean_e1, clean_e2, channel_crossings, e1_peaks_m, e1_troughs_m, e2_peaks_m, e2_troughs_m


def plot_episode(e1_peaks, e2_peaks, e1_troughs, e2_troughs, e1_uvd, e2_uvd, episode):
    start_art = []  # only update with values when another bit of code is run for some reason
    end_art = []  # only update with values when another bit of code is run for some reason
    fig, ax = plt.subplots()
    ax.plot(e1_uvd, linewidth=0.8, color='navy', label='EOG1')
    ax.plot(e2_uvd, linewidth=0.8, color='mediumblue', label='EOG2')
    ax.plot(e1_peaks, e1_uvd[e1_peaks], 'o', color='darkred', label='EOG1 Peaks')
    ax.plot(e2_peaks, e2_uvd[e2_peaks], 'o', color='darkorange', label='EOG2 Peaks')
    ax.plot(e1_troughs, e1_uvd[e1_troughs], 'D', color='darkorange', label='EOG1 Troughs')
    ax.plot(e2_troughs, e2_uvd[e2_troughs], 'D', color='darkred', label='EOG2 Troughs')
    plt.title(label=['Mark *ARTEFACTS* (drag cursor): Episode', episode])
    ax.legend(loc='upper right')
    ax.set(facecolor='whitesmoke')
    plt.xlabel('Samples', fontsize=10)
    plt.ylabel('Voltage (uV)', fontsize=10)

    def onselect(vmin, vmax):
        start_art.append(vmin)
        end_art.append(vmax)

    rectprops = dict(facecolor='red', alpha=0.5)
    span = mwidgets.SpanSelector(ax, onselect, 'horizontal',
                                 rectprops=rectprops, span_stays=True)
    plt.show(block=True)
    return start_art, end_art


def remove_art_peaks(e1_peaks, e2_peaks, e1_troughs, e2_troughs, start_art, end_art, episode):
    true_e1_p = e1_peaks.copy()
    true_e1_t = e1_troughs.copy()
    true_e2_p = e2_peaks.copy()
    true_e2_t = e2_troughs.copy()
    rejected_e1_p = []
    rejected_e1_t = []
    rejected_e2_p = []
    rejected_e2_t = []
    if len(start_art) > 0:
        for index in range(len(start_art)):
            for idxp, valp in enumerate(e1_peaks):
                if start_art[index] <= e1_peaks[idxp] <= end_art[index]:
                    rejected_e1_p.append(valp), true_e1_p.remove(valp)
            for idxt, valt in enumerate(e1_troughs):
                if start_art[index] <= valt <= end_art[index]:
                    rejected_e1_t.append(valt), true_e1_t.remove(valt)
            for idxy, valy in enumerate(e2_peaks):
                if start_art[index] <= valy <= end_art[index]:
                    rejected_e2_p.append(valy), true_e2_p.remove(valy)
            for idxz, valz in enumerate(e2_troughs):
                if start_art[index] <= valz <= end_art[index]:
                    rejected_e2_t.append(valz), true_e2_t.remove(valz)
        total_peaks = len(e1_peaks) + len(e1_troughs) + len(e2_peaks) + len(e2_troughs)
        total_rejected = len(rejected_e1_p) + len(rejected_e1_t) + len(rejected_e2_p) + len(rejected_e2_t)
        percent_rejected = (total_rejected / total_peaks) * 100
        print('episode:', episode, '|  Analysis completed |  total peaks:', total_peaks, '|  total rejected due to artefact:', total_rejected, '|  percent rejected:',
              percent_rejected)
    elif len(start_art) == 0:
        true_e1_p = e1_peaks.copy()
        true_e1_t = e1_troughs.copy()
        true_e2_p = e2_peaks.copy()
        true_e2_t = e2_troughs.copy()
        total_peaks = len(e1_peaks) + len(e1_troughs) + len(e2_peaks) + len(e2_troughs)
        total_rejected = 0
        percent_rejected = 0
        print('episode:', episode, '|  Analysis completed |  total peaks:', total_peaks, '|  total rejected due to artefact:', total_rejected, '|  percent rejected:',
              percent_rejected)
    return true_e1_p, true_e1_t, true_e2_p, true_e2_t


def signal_derivative(signal):
    dx = 1
    y = signal
    dy = np.diff(y) / dx
    first_derivative = scipy.stats.zscore(dy)
    return first_derivative


def slope(x1, y1, x2, y2):
    grad_by_sample = (y2 - y1) / (x2 - x1)
    grad_by_sf = (y2 - y1) / ((x2 / 256) - (x1 / 256))
    return grad_by_sample, grad_by_sf


def match_crossing_to_peaks(peaks_troughs, channel_crossings):
    """ this function takes the peak and channel crossing indexes and identifies the channel crossings immediately
    to the left and right of the peak. We therefore find the beginning/end of the REMs movement.
    This function also calculates the difference in sample# between the start crossing & peak, peak & end crossing
   """
    rems_start_crossings = []
    rems_end_crossings = []
    start_to_peak_distances = []
    peak_to_end_distances = []
    for peak in peaks_troughs:
        nearest_crossing = find_nearest(channel_crossings, peak)
        nearest_cross_idx = (np.where(channel_crossings == nearest_crossing))[0]
        nearest_cross_idx = int(nearest_cross_idx)
        last_crossing = len(channel_crossings) - 1
        if nearest_crossing > peak:  # if peak index precedes crossing index
            start_cross_idx = channel_crossings[nearest_cross_idx - 1]
            end_cross_idx = nearest_crossing
        elif channel_crossings[nearest_cross_idx] == channel_crossings[last_crossing]:
            start_cross_idx = nearest_crossing
            end_cross_idx = last_crossing
        else:
            start_cross_idx = nearest_crossing
            end_cross_idx = channel_crossings[nearest_cross_idx + 1]
        rems_start_crossings.append(start_cross_idx)
        rems_end_crossings.append(end_cross_idx)
        start_to_peak_diff = peak - start_cross_idx
        start_to_peak_distances.append(start_to_peak_diff)
        peak_to_end_diff = end_cross_idx - peak
        peak_to_end_distances.append(peak_to_end_diff)

    return rems_start_crossings, rems_end_crossings, start_to_peak_distances, peak_to_end_distances


def epoch_data(signal, peaks_troughs, invert):
    if invert == False:
        # epoch data around matched peaks or troughs
        signal_events_raw = nk.epochs_create(signal, peaks_troughs, sampling_rate=256, epochs_start=-2,
                                             epochs_end=2)  # contains OG index info
        signal_events = nk.epochs_to_array(signal_events_raw)
        epochs = signal_events.transpose()
    else:
        invert_signal = signal * -1  # invert signal to make troughs become peaks
        signal_events_raw = nk.epochs_create(invert_signal, peaks_troughs, sampling_rate=256, epochs_start=-2,
                                             epochs_end=2)  # contains OG index info
        signal_events = nk.epochs_to_array(signal_events_raw)
        epochs = signal_events.transpose()
    return epochs


def find_landmarks(epochs, peak_troughs, rems_start_crossings, rems_end_crossings, start_to_peak_distances,
                   peak_to_end_distances):
    peak_values = []
    loc_min_peak_diff = []
    rems_local_minima = []
    true_start_peak_diff = []
    count = 0
    for epoch in epochs:
        peak_idx = 256 * 2  # index of peak - is always SF*2 as we have 2s buffer either side
        peak_val = epoch[peak_idx]  # value of peak
        peak_values.append(peak_val)

        # find local minima before crossing
        reverse = epoch[::-1]
        if start_to_peak_distances[count] < (256 * 2):
            ch_crossing_idx = peak_idx + start_to_peak_distances[count]
            reverse_crop = reverse[ch_crossing_idx:]
            # find closest small number to start of cropped data segment
            loc_minima_idx = \
                np.where((reverse_crop[1:-1] < reverse_crop[0:-2]) * (reverse_crop[1:-1] < reverse_crop[2:]))[0] + 1
            if loc_minima_idx.size == 0:  # for if lowest value is at last value of the cropped data segment
                loc_minima_index = len(reverse_crop)
                loc_minima_index = np.array([loc_minima_idx])
            else:
                loc_minima_index = loc_minima_idx[0]
            # min_to_peak_diff = peak_idxs[count] + (1024(if sf=512) - (loc_minima_index + ch_crossing_idx))
            cross_to_min = loc_minima_index
            min_to_peak = cross_to_min + ch_crossing_idx
            min_to_peak_diff = min_to_peak - 512
        else:
            min_to_peak_diff = 'NaN'
            true_start = 'NaN'

        true_start_peak_diff.append(min_to_peak_diff)
        loc_min_peak_diff.append(min_to_peak_diff)

    for peak in range(len(peak_troughs)):
        locmin = peak_troughs[peak] - loc_min_peak_diff[peak]
        rems_local_minima.append(locmin)

    return count, loc_min_peak_diff, rems_local_minima


def rems_characteristics(episode_count, rem_episodes, peaks_troughs, rems_start_crossings, rems_end_crossings,
                         rems_local_minima, e1_peaks, e2_peaks, e1_troughs, e2_troughs, eye, direction):
    # calculate values
    ep = rem_episodes[episode_count]
    ep = np.ndarray.flatten(ep)
    ep = scipy.signal.detrend(ep, type='linear', overwrite_data=True)  # detrend data (linear)
    # create z_scored cleaned data
    z = scipy.stats.zscore(ep)
    clean_z = nk.eog_clean(z, sampling_rate=256, method='nk')
    clean_epochs_z = nk.epochs_create(clean_z, peaks_troughs, 256, epochs_start=-2,
                                      epochs_end=2)  # contains OG index info
    cleaned_events_z = nk.epochs_to_array(clean_epochs_z)  # cleaned with agarwal2019 method
    cleaned_epochs_z = cleaned_events_z.transpose()
    # create cleaned data of original amplitudes
    clean = nk.eog_clean(ep, sampling_rate=256, method='nk')
    clean_epochs = nk.epochs_create(clean, peaks_troughs, 256, epochs_start=-2,
                                    epochs_end=2)  # contains OG index info
    cleaned_events = nk.epochs_to_array(clean_epochs)  # cleaned with agarwal2019 method
    cleaned_epochs = cleaned_events.transpose()
    # first calculate features of interest:
    peak_idx = 512
    amplitude_abs = []
    amplitude_rel = []
    gradient_abs = []
    gradient_cross = []
    gradient_decay = []
    duration_abs = []
    duration_cross = []
    duration_decay = []
    prior_peak_distance = []
    post_peak_distance = []

    # calculate distance between peaks
    all_peaks = e1_peaks + e2_peaks + e1_troughs + e2_troughs
    sorted_all_peaks = sorted(all_peaks)
    sorted_all_peaks_dict = dict.fromkeys(sorted_all_peaks)
    sorted_all_peaks = list(sorted_all_peaks_dict)

    for index, value in enumerate(peaks_troughs):
        last_index = len(peaks_troughs) - 1
        nearest_peak = find_nearest(sorted_all_peaks, value)
        peak_in_list = (np.where(sorted_all_peaks == nearest_peak))[0]
        peak_in_list = np.ndarray.flatten(peak_in_list)
        peak_in_list = int(peak_in_list)
        if peaks_troughs[index] == peaks_troughs[0]:
            prior_peak_val = 'NaN'
            prior_peak_distance.append(prior_peak_val)
        else:
            prior_peak_val = sorted_all_peaks[peak_in_list - 1]
            prior_peak_s = (value - prior_peak_val) / 256
            prior_peak_distance.append(prior_peak_s)

        if peaks_troughs[index] == peaks_troughs[last_index]:
            post_peak_val = 'NaN'
            post_peak_distance.append(post_peak_val)
        else:
            post_peak_val = sorted_all_peaks[peak_in_list + 1]
            post_peak_s = (post_peak_val - value) / 256
            post_peak_distance.append(post_peak_s)

    # calculate durations (in samples & seconds)
    duration_samps = []
    dc_samps = []
    dd_samps = []
    for value in range(len(peaks_troughs)):
        duration_samps.append(peaks_troughs[value] - rems_local_minima[value])
        duration_abs.append((peaks_troughs[value] - rems_local_minima[value]) / 256)
        dc_samps.append((peaks_troughs[value] - rems_start_crossings[value]))
        duration_cross.append((peaks_troughs[value] - rems_start_crossings[value]) / 256)
        dd_samps.append((rems_end_crossings[value] - peaks_troughs[value]))
        duration_decay.append((rems_end_crossings[value] - peaks_troughs[value]) / 256)
        amp_abs = cleaned_epochs[value][peak_idx]
        amp_abs_mv = abs(amp_abs * 1000000)
        amplitude_abs.append(amp_abs_mv)
        amp_rel = abs(cleaned_epochs_z[value][peak_idx])
        amplitude_rel.append(amp_rel)
    # calculate gradients
    count = 0
    for epoch in cleaned_epochs_z:
        # local minimum to peak rise gradient
        lmin_to_peak = peak_idx - duration_samps[count]
        x1 = lmin_to_peak
        y1 = epoch[lmin_to_peak]
        x2 = peak_idx
        y2 = epoch[peak_idx]
        abs_rise_by_sample, abs_rise_by_sf = slope(x1, y1, x2, y2)
        abs_rise_by_sf_pos = abs(abs_rise_by_sf)
        gradient_abs.append(abs_rise_by_sf_pos)

        # channel crossing to peak rise gradient
        cross_to_peak = peak_idx - dc_samps[count]
        x1 = cross_to_peak
        y1 = epoch[cross_to_peak]
        x2 = peak_idx
        y2 = epoch[peak_idx]
        cr_rise_by_sample, cr_rise_by_sf = slope(x1, y1, x2, y2)
        cr_rise_by_sf_pos = abs(cr_rise_by_sf)
        gradient_cross.append(cr_rise_by_sf_pos)

        # peak to end ch_crossing decay gradient
        peak_to_end = dd_samps[count] - peak_idx
        x1 = peak_to_end
        y1 = epoch[peak_to_end]
        x2 = peak_idx
        y2 = epoch[peak_idx]
        decay_by_sample, decay_by_sf = slope(x1, y1, x2, y2)
        gradient_decay_neg = -abs(decay_by_sf)
        gradient_decay.append(gradient_decay_neg)
        count += 1

    rems_df = rems_dataframe(episode_count, eye, direction, peaks_troughs, amplitude_abs, amplitude_rel, gradient_abs,
                             gradient_cross,
                             gradient_decay, duration_abs, duration_cross, duration_decay, prior_peak_distance,
                             post_peak_distance)
    return rems_df


def rems_dataframe(episode_count, eye, direction, peaks_troughs, amplitude_abs, amplitude_rel, gradient_abs,
                   gradient_cross, gradient_decay, duration_abs, duration_cross, duration_decay,
                   prior_peak_distance, post_peak_distance):
    eye_list = [eye] * len(peaks_troughs)
    direction_list = [direction] * len(peaks_troughs)
    episode_count_list = [episode_count] * len(peaks_troughs)
    rems_df = pd.DataFrame(data=
                           list(zip(episode_count_list, eye_list, direction_list, peaks_troughs, amplitude_abs,
                                    amplitude_rel,
                                    gradient_abs,
                                    gradient_cross, gradient_decay, duration_abs, duration_cross, duration_decay,
                                    prior_peak_distance, post_peak_distance)),
                           columns=['Episode', 'Eye', 'Direction', 'Peak Index', 'Absolute Amplitude (uV)',
                                    'Relative Amplitude', 'Absolute Gradient', 'Cross Gradient', 'Decay Gradient',
                                    'Absolute Duration (s)', 'Cross Duration (s)', 'Decay Duration (s)',
                                    'Prior Peak Distance (s)', 'Post Peak Distance (s)'])
    return rems_df


def rems_clusters(episode, cluster_intervals, mean_intraclus_ipi,
                  cluster_peak_density, cluster_duration):
    episode_count_list = [episode] * len(cluster_intervals)
    cluster_count = list(range(0, len(cluster_intervals)))
    rems_cluster = pd.DataFrame(data=
                                list(zip(episode_count_list, cluster_count, cluster_intervals,
                                         mean_intraclus_ipi, cluster_peak_density, cluster_duration)),
                                columns=['Episode', 'Cluster Number',
                                         'Cluster Intervals (s)', 'Avg Intracluster Interval (s)',
                                         'Cluster Peak Density', 'Cluster Duration (s)'])
    return rems_cluster


def phasic_tonic_detections(e1_peaks, e2_peaks, e1_troughs, e2_troughs, signal, episode, start_art, end_art):
    # calculate inter-peak-interval (3s clustering defined by tononi 2021)
    combined_peaks = e1_peaks + e1_troughs
    combined_peaks.sort()
    ipi = [combined_peaks[n] - combined_peaks[n - 1] for n in
           range(1, len(combined_peaks))]  # output should be list of the difference values
    single_events = []
    clusters = []
    cluster_starts = []
    cluster_ends = []
    count = 0
    # peaks occurring within 3 seconds of each other
    for index, value in enumerate(combined_peaks):
        if value == combined_peaks[0] and combined_peaks[index + 1] >= value + 768:
            single_events.append(value)
        elif value == combined_peaks[0] and combined_peaks[index + 1] <= value + 768:
            clusters.append(value)  # this would be the start of a cluster
            cluster_starts.append(value)
        elif index == len(combined_peaks) - 1 and combined_peaks[index - 1] >= value + 768:
            single_events.append(value)
        elif index == len(combined_peaks) - 1 and combined_peaks[index - 1] <= value + 768:
            clusters.append(value)
            cluster_ends.append(value)
        elif value >= (combined_peaks[index - 1] + 768) and combined_peaks[index + 1] <= value + 768:
            clusters.append(value)  # this is the start of a cluster
            cluster_starts.append(value)
        elif value <= (combined_peaks[index - 1] + 768) and combined_peaks[index + 1] <= value + 768:
            clusters.append(value)  # these are the middle bits of a cluster
        elif value <= (combined_peaks[index - 1] + 768) and combined_peaks[index + 1] >= value + 768:
            clusters.append(value)  # this is the end of a cluster
            cluster_ends.append(value)
        elif value >= (combined_peaks[index - 1] + 768) and combined_peaks[index + 1] >= value + 768:
            single_events.append(value)

    #  clusters_ipi = [clusters[n] - clusters[n - 1] for n in range(1, len(clusters))]  # sanity check
    #  single_events_ipi = [single_events[n] - single_events[n - 1] for n in range(1, len(single_events))]

    # create list of clusters
    list_of_clusters = []
    for index in range(len(cluster_starts)):
        cluster_list = []
        for i, value in enumerate(combined_peaks):
            if cluster_starts[index] <= value <= cluster_ends[index]:
                cluster_list.append(value)
            else:
                pass
        list_of_clusters.append(cluster_list)

        # calculate some cluster characteristics
    cluster_peak_density = []  # in seconds!!!
    cluster_duration = []  # in seconds
    for cluster in list_of_clusters:
        num_peaks = len(cluster)
        cluster_start_value = cluster[0]
        cluster_end_value = cluster[num_peaks - 1]
        num_samples = cluster_end_value - cluster_start_value
        cluster_duration.append((num_samples / 256))
        seconds = num_samples / 256
        density = num_peaks / seconds
        cluster_peak_density.append(density)

        # calculate ipi for within each cluster
    intracluster_ipi = []
    for cluster in list_of_clusters:
        difference = [cluster[n] - cluster[n - 1] for n in range(1, len(cluster))]
        intracluster_ipi.append(difference)
    mean_intraclus_ipi = []
    for cluster in intracluster_ipi:
        mean = statistics.mean(cluster)
        mean_s = mean / 256  # mean ipi in seconds
        mean_intraclus_ipi.append(mean_s)

    # find out time between cluster start & ends
    list_of_clusters_cp = list_of_clusters.copy()
    cluster_intervals = []
    for index in range(len(list_of_clusters_cp)):
        if index < len(list_of_clusters_cp) - 1:
            list_last_idx = len(list_of_clusters_cp[index]) - 1
            difference = list_of_clusters_cp[index + 1][0] - list_of_clusters_cp[index][list_last_idx]
            difference_s = difference / 256  # difference in seconds
            cluster_intervals.append(difference_s)
        else:
            pass

    # find out percentage of phasic v tonic sleep
    # populate remgrams: tonic rem = 0, phasic rem = 1, artefact = 5
    remgram = [0] * len(signal)
    # buffers = 4s either side of start/end of cluster/single event
    for index, value in enumerate(cluster_starts):
        if value > 1024:
            buffer_start = value - 1024
        else:
            buffer_start = 0
        end_threshold = len(signal) - 1024
        if cluster_ends[index] > end_threshold:
            buffer_end = len(signal)
        else:
            buffer_end = cluster_ends[index] + 1024
        diff = buffer_end - buffer_start
        cluster_ones = [1] * diff
        replacement_indexes = range(buffer_start, buffer_end, 1)
        for (i, replacement) in zip(replacement_indexes, cluster_ones):
            remgram[i] = replacement
    for index, value in enumerate(single_events):
        if value > 1024:
            buffer_sing_start = value - 1024
        else:
            buffer_sing_start = 0
        end_threshold = len(signal) - 1024
        if value > end_threshold:
            buffer_sing_end = len(signal)
        else:
            buffer_sing_end = value + 1024
        diff = buffer_sing_end - buffer_sing_start
        cluster_sing_ones = [1] * diff
        replacement_sing_indexes = range(buffer_sing_start, buffer_sing_end, 1)
        for (i, replacement) in zip(replacement_sing_indexes, cluster_sing_ones):
            remgram[i] = replacement
    if len(start_art) > 0:
        for index, value in enumerate(start_art):
            int_end = int(end_art[index])
            int_start = int(start_art[index])
            diff = int_end - int_start
            cluster_art = [5] * diff
            replacement_art_indexes = range(int_start, int_end, 1)
            for (i, replacement) in zip(replacement_art_indexes, cluster_art):
                remgram[i] = replacement
    else:
        pass
    # calculate % spent phasic rem v tonic rem
    total_samples = len(signal)
    total_duration = total_samples / 256  # seconds
    tonic = remgram.count(0)
    total_ton_dur = tonic / 256  # seconds
    phasic = remgram.count(1)
    total_phas_dur = phasic / 256  # seconds
    tonic_percentage = (tonic / total_samples) * 100
    phasic_percentage = (phasic / total_samples) * 100
    if len(start_art) > 0:
        art = remgram.count(5)
        art_percentage = (art / total_samples) * 100
        total_art_dur = art / 256  # seconds
    else:
        art = 0
        art_percentage = 0
        total_art_dur = 0

    rems_cluster = rems_clusters(episode, cluster_intervals, mean_intraclus_ipi, cluster_peak_density,
                                 cluster_duration)

    #    rems_micros = rems_microstates(episode, tonic_percentage, phasic_percentage)

    return remgram, rems_cluster, tonic_percentage, phasic_percentage, art_percentage, total_duration, total_ton_dur, \
           total_phas_dur, total_art_dur


def initialise_tp_micro(episode, tonic_percentage, phasic_percentage, episode_list, tonic_percentage_list,
                        phasic_percentage_list, td_list, ttd_list, tpd_list, total_duration, total_ton_dur,
                        total_phas_dur, ap_list, tad_list, art_percentage, total_art_dur):
    episode_list.append(episode)
    phasic_percentage_list.append(phasic_percentage)
    tonic_percentage_list.append(tonic_percentage)
    td_list.append(total_duration)
    ttd_list.append(total_ton_dur)
    tpd_list.append(total_phas_dur)
    ap_list.append(art_percentage)
    tad_list.append(total_art_dur)
    return episode_list, tonic_percentage_list, phasic_percentage_list, td_list, ttd_list, tpd_list, ap_list, tad_list


def rems_microstates(ep_list, tp_list, pp_list, ap_list, td_list, ttd_list, tpd_list, tad_list):
    rems_microstates = pd.DataFrame(data=(zip(ep_list, tp_list, pp_list, ap_list, td_list, ttd_list, tpd_list,
                                              tad_list)),
                                    columns=['Episode', 'Tonic percentage', 'Phasic Percentage',
                                             'Artefact percentage', 'Total Duration(s)', 'Total Tonic Duration(s)',
                                             'Total Phasic Duration (s)', 'Artefact Duration (s)'])
    return rems_microstates


def calculate_microstates():
    ep_list = []  # initalise episode list
    pp_list = []  # initialise phasic percentage list
    tp_list = []  # initialise tonic percentage list
    ap_list = []  # initialise artefact percentage list
    td_list = []  # initialise total duration list
    ttd_list = []  # initialise total ton. duration list
    tpd_list = []  # initalise total phas. duration list
    tad_list = []  # initialise total art duration list
    rems_microstates_df = rems_microstates(ep_list, tp_list, pp_list, ap_list, td_list, ttd_list, tpd_list, tad_list)
    return rems_microstates_df
