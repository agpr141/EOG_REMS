"""
Script to check manual sleep scoring of EEG files is correct before EOG analysis.

author: @agpr141
last update: 23/11/21
"""

# import modules
from pathlib import Path
import os
import mne
import datetime
import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
import scipy

# set paths
path = Path('Y:/22qEEG/E018-5-2-1')  # define path to participant folder
os.chdir(path)  # change working directory to path
hypno = 'scoring_outputs/E018-5-2-1_scoring_info_novid.csv'  # define path to hypnogram
EEG = 'exported_data/E018-5-2-1_sleep_EEG_PREP.edf'  # define path to EEG-PREP preprocessed file

# load in EEG edf
EEG_sig = mne.io.read_raw_edf(EEG, exclude=['F5', 'FP1', 'FP2' 'F6', 'F10', 'F7', 'FC5', 'F3', 'F1', 'F2',
                                            'FC6', 'T9', 'FT7', 'FC3', 'FC1', 'FC2', 'FC4', 'FT8', 'T10', 'T7',
                                            'C5', 'C3', 'C1', 'C2', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP1',
                                            'CP2', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1', 'P2', 'P4', 'P6',
                                            'P8', 'AF4', 'P9', 'PO3', 'O1', 'PO4', 'P10', 'Oz', 'AF3', 'Fz',
                                            'FCz', 'Cz', 'Pz', 'POz', 'FP2', 'F6'], preload=True)  # F4, C4, O2, F8, F9 (eyes)

# pick scoring montage (AASM: F4, C4, O2 + F8 & F9 for right & left EOG channels respectively)
Final = EEG_sig.pick_types(include=["F4", "C4", "O2", "F8", "F9"])

# reorder and turn Final from {RawEDF} to nparray Preprocessed:
FinalN = Final.reorder_channels(ch_names=["F4", "C4", "O2", "F8", "F9"])
Preprocessed, times = FinalN[:, :]

# run SLEEP --> load in hypnogram using GUI
from visbrain.gui import Sleep
Sleep(data=Preprocessed, channels=["F4", "C4", "O2", "F8", "F9"], sf=128,
      use_mne=True).show()
