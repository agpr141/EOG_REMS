"""
script to automatically identify eye movements during a pre-specified REM episode & analyse
use this script if one of your episodes did not get analysed due to a script bug
author: @agpr141
last updated: 01/12/21
"""
# ----------------------------------------------------------------------------------------------------------------------
# SECTION 1  - import modules, specify file paths, align hypnogram to EOG file & extract REM periods

# import modules - pip install these!
import os
import sys
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
from matplotlib.widgets import Button
import warnings

warnings.filterwarnings("ignore")

# specify paths
path = Path('Y:/22qEEG/E004-1-1-1')  # define path to participant folder
os.chdir(path)  # change working directory to path
hypnogram = 'scoring_outputs/E004-1-1-1_scoring_info_novid.csv'  # define hypnogram path
EEG = 'exported_data/E004-1-1-1_sleep_EEG_PREP.edf'  # define EEG data path (for hypnogram alignment)
EOG = 'exported_data/E004-1-1-1_PSG.edf'  # define EOG data path
sampling_freq = 256  # define sampling frequency for EOG data
episode_to_analyse = 1  # define the episode you want to analyse here

# preprocess data & extract rem periods for each eog channel
rem_episodes_e1, rem_episodes_e2, rem_timings = extract_rem_episodes(EEG, EOG, hypnogram)

# ----------------------------------------------------------------------------------------------------------------------
# SECTION 2  - manual identification of peaks, mark bad episodes, mark artefacts, calculate rem, cluster & microstate characteristics

# initialise variables to append to later
sys.stdout = open('automatic_processing_output_single_episode{}.txt'.format(episode_to_analyse), 'w')
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
tpd_list = []  # initalise total phasic duration list
tad_list = []  # initialise total art duration list
rems_microstates_df = pandas.DataFrame()

clean_e1, clean_e2, channel_crossings, e1_peaks, e1_troughs, e2_peaks, e2_troughs, e1_uvd, e2_uvd = \
    matched_peaks_detection(episode_to_analyse, rem_episodes_e1, rem_episodes_e2)

start_art, end_art, bad_episode_list, good_episode_list = mark_bad_or_artefact(e1_peaks, e2_peaks, e1_troughs,
                                                                               e2_troughs, e1_uvd, e2_uvd,
                                                                               episode_to_analyse, bad_episode_list,
                                                                               good_episode_list)

e1_peaks, e1_troughs, e2_peaks, e2_troughs = remove_art_peaks(e1_peaks, e2_peaks, e1_troughs, e2_troughs,
                                                              start_art, end_art, episode_to_analyse)
# analyse peaks/troughs from each channel
rems_df_e1p = rems_analyse(clean_e1, e1_peaks, channel_crossings, 'Left', 'Left', e1_peaks, e1_troughs,
                           e2_peaks, e2_troughs, rem_episodes_e1, episode_to_analyse, invert=False)

rems_df_e1t = rems_analyse(clean_e1, e1_troughs, channel_crossings, 'Left', 'Right', e1_peaks, e1_troughs,
                           e2_peaks, e2_troughs, rem_episodes_e1, episode_to_analyse, invert=True)

rems_df_e2p = rems_analyse(clean_e2, e2_peaks, channel_crossings, 'Right', 'Right', e1_peaks, e1_troughs,
                           e2_peaks, e2_troughs, rem_episodes_e2, episode_to_analyse, invert=False)

rems_df_e2t = rems_analyse(clean_e2, e2_troughs, channel_crossings, 'Right', 'Left', e1_peaks, e1_troughs,
                           e2_peaks, e2_troughs, rem_episodes_e2, episode_to_analyse, invert=True)

rems_df = rems_df.append([rems_df_e1p, rems_df_e1t, rems_df_e2p, rems_df_e2t])  # compile rem characteristics

remgram, rems_clusters, tonic_percentage, phasic_percentage, art_percentage, total_duration, total_ton_dur, \
total_phas_dur, total_art_dur = phasic_tonic_detections(e1_peaks, e2_peaks, e1_troughs, e2_troughs, clean_e1,
                                                        episode_to_analyse, start_art, end_art)

rems_cluster_df = rems_cluster_df.append(rems_clusters)  # compile cluster characteristics

ep_list, tp_list, pp_list, td_list, ttd_list, tpd_list, ap_list, tad_list = initialise_tp_micro(
    episode_to_analyse, tonic_percentage, phasic_percentage, ep_list, tp_list, pp_list, td_list, ttd_list, tpd_list,
    total_duration, total_ton_dur, total_phas_dur, ap_list, tad_list, art_percentage, total_art_dur)

# compile microstate characteristics
rems_microstates_df = rems_microstates(ep_list, tp_list, pp_list, ap_list, td_list, ttd_list, tpd_list, tad_list)
sys.stdout.close()

# ----------------------------------------------------------------------------------------------------------------------
# SECTION 3  - save outputs to csv files

# save individual eye movement, cluster & microstate characteristics to .csv file format into participant folder
rems_df.to_csv(path_or_buf='automatic_rems_characteristics_episode{}.csv'.format(episode_to_analyse), index_label='Peak Number in Ep')
rems_cluster_df.to_csv(path_or_buf='automatic_rems_clusters_episode{}.csv'.format(episode_to_analyse), index=False)
rems_microstates_df.to_csv(path_or_buf='automatic_rems_microstates_episode{}.csv'.format(episode_to_analyse), index=False)
