"""
script to automatically identify eye movements during REM sleep & analyse
author: @agpr141]
last updated: 01/12/12
"""
# ----------------------------------------------------------------------------------------------------------------------
# SECTION 1  - import modules, specify file paths, align hypnogram to EOG file & extract REM periods

# import modules - pip install if missing
import os
from pathlib import Path
import mne
import numpy as np
import pandas
from matplotlib import pyplot as plt
import scipy
import neurokit2 as nk
import statistics
import matplotlib.widgets as mwidgets
from rems_analyse import *
from rems_functions import *
import warnings
warnings.filterwarnings("ignore")

# specify paths
path = Path('Y:/22qEEG/E006-1-1-1')  # define path to file
os.chdir(path)  # change working directory to path
hypnogram = 'scoring_outputs/E006-1-1-1_scoring_info_vid.csv'  # define hypnogram path
EEG = 'exported_data/E006-1-1-1_sleep_EEG_PREP.edf'  # define EEG data path (for hypnogram alignment)
EOG = 'exported_data/E006-1-1-1_PSG.edf'  # define EOG data path
sampling_freq = 256  # define sampling frequency for EOG data

# preprocess data & extract rem periods for each eog channel
rem_episodes_e1, rem_episodes_e2, rem_timings = extract_rem_episodes(EEG, EOG, hypnogram)

# ----------------------------------------------------------------------------------------------------------------------
# SECTION 2  - manual identification of peaks, mark bad episodes, mark artefacts, calculate rem, cluster & microstate characteristics

# initialise variables to append to later
sys.stdout = open('automatic_processing_output.txt', 'w')
episode_count = 0
rems_df = pandas.DataFrame()
rems_cluster_df = pandas.DataFrame()
episode_list = []
bad_episode_list = []
good_episode_list = []
ep_list = []  # initalise episode list
pp_list = []  # initialise phasic percentage list
tp_list = []  # initialise tonic percentage list
ap_list = []  # initialise artefact percentage list
td_list = []  # initialise total duration list
ttd_list = []  # initialise total ton. duration list
tpd_list = []  # initalise total phas. duration list
tad_list = []  # initialise total art duration list
rems_microstates_df = pandas.DataFrame()

# automatic peak detection for each REM episode - you mark 'bad episode' or artefacts. automatic analysis follows
for episode in range(len(rem_episodes_e1)):
    try:
        clean_e1, clean_e2, channel_crossings, e1_peaks, e1_troughs, e2_peaks, e2_troughs = \
            matched_peaks_detection(episode, rem_episodes_e1, rem_episodes_e2)

        start_art, end_art, bad_episode_list, good_episode_list = mark_bad_or_artefact(e1_peaks, e2_peaks, e1_troughs,
                                                                                       e2_troughs, clean_e1, clean_e2,
                                                                                       episode, bad_episode_list,
                                                                                       good_episode_list)
        episode_count += 1
        # if episode is marked 'bad' it is excluded from further analysis
        if episode in bad_episode_list:
            print('episode', episode, 'is a bad episode. no analysis done.')
        # if episode is marked 'good' automatic analysis follows.
        else:
            e1_peaks, e1_troughs, e2_peaks, e2_troughs = remove_art_peaks(e1_peaks, e2_peaks, e1_troughs, e2_troughs, start_art,
                                                                          end_art, episode)

            rems_df_e1p = rems_analyse(clean_e1, e1_peaks, channel_crossings, 'Left', 'Left', e1_peaks, e1_troughs, e2_peaks, e2_troughs,
                             rem_episodes_e1, episode, invert=False)

            rems_df_e1t = rems_analyse(clean_e1, e1_troughs, channel_crossings, 'Left', 'Right', e1_peaks, e1_troughs, e2_peaks,
                             e2_troughs, rem_episodes_e1, episode, invert=True)

            rems_df_e2p = rems_analyse(clean_e2, e2_peaks, channel_crossings, 'Right', 'Right', e1_peaks, e1_troughs, e2_peaks,
                             e2_troughs, rem_episodes_e2, episode, invert=False)

            rems_df_e2t = rems_analyse(clean_e2, e2_troughs, channel_crossings, 'Right', 'Left', e1_peaks, e1_troughs, e2_peaks,
                             e2_troughs, rem_episodes_e2, episode, invert=True)

            rems_df = rems_df.append([rems_df_e1p, rems_df_e1t, rems_df_e2p, rems_df_e2t])

            remgram, rems_clusters, tonic_percentage, phasic_percentage, art_percentage, total_duration, total_ton_dur, \
            total_phas_dur, total_art_dur = phasic_tonic_detections(e1_peaks, e2_peaks, e1_troughs, e2_troughs, clean_e1,
                                                                    episode, start_art, end_art)

            rems_cluster_df = rems_cluster_df.append(rems_clusters)

            ep_list, tp_list, pp_list, td_list, ttd_list, tpd_list, ap_list, tad_list = initialise_tp_micro(
                episode, tonic_percentage, phasic_percentage, ep_list, tp_list, pp_list, td_list, ttd_list, tpd_list,
                total_duration, total_ton_dur, total_phas_dur, ap_list, tad_list, art_percentage, total_art_dur)

            episode_count += 1

        rems_microstates_df = rems_microstates(ep_list, tp_list, pp_list, ap_list, td_list, ttd_list, tpd_list, tad_list)

    except:
        print('episode', episode, ': something went wrong in the code. No analysis done. Analyse independently')
        episode_count += 1
sys.stdout.close()

# ----------------------------------------------------------------------------------------------------------------------
# SECTION 3  - save outputs to csv files

# save individual eye movement, cluster & microstate characteristics to .csv file format into participant folder
rems_df.to_csv(path_or_buf='automatic_rems_characteristics.csv', index_label='Peak Number in Ep')
rems_cluster_df.to_csv(path_or_buf='automatic_rems_clusters.csv', index=False)
rems_microstates_df.to_csv(path_or_buf='automatic_rems_microstates.csv', index=False)
