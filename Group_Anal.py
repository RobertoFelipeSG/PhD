import numpy as np
import scipy
import mne
import matplotlib
import matplotlib.pyplot as plt

from TimeFreqAna import *
from Group_TF import *

# from GrangerCaus import *
# from Group_Gr import *

from SourcesReco import *
from Group_Src import *

# Channels of interest
chans_all = ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'FZ', 'CZ', 'PZ', 'OZ', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9', 'TP10', 'POZ', 'ECG', 'F1', 'F2', 'C1', 'C2', 'P1', 'P2', 'AF3', 'AF4', 'FC3', 'FC4', 'CP3', 'CP4', 'PO3', 'PO4', 'F5', 'F6', 'C5', 'C6', 'P5', 'P6', 'AF7', 'AF8', 'FT7', 'FT8', 'TP7', 'TP8', 'PO7', 'PO8', 'FT9', 'FT10', 'FPZ', 'FCZ']

chans_lh = ['FP1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T7', 'P7',# 'FZ', 'CZ', 'PZ', 'OZ', 
            'FC1', 'CP1', 'FC5', 'CP5', 'TP9', 'POZ', 'F1', 'C1', 'P1', 'AF3', 
            'FC3', 'CP3', 'PO3', 'F5', 'C5', 'P5', 'AF7', 'FT7', 'TP7', 'PO7', 'FT9'] 
#             'FPZ', 'FCZ']
        
chans_rh = ['FP2', 'F4', 'C4', 'P4', 'O2', 'F8', 'T8', 'P8', 'FZ', 'CZ', 'PZ', 'OZ',
            'FC2', 'CP2', 'FC6', 'CP6', 'TP10', 'POZ', 'F2', 'C2', 'P2',
             'CP4', 'PO4', 'F6', 'C6', 'P6', 'AF8', 'FT8', 'TP8', 'PO8' , 'FT10', 
             'FCZ']#'FC4', 'AF4', 'FPZ',

chans_r = ['CP4', 'CP2', 'CPZ', 'P6', 'P4', 'P2', 'PZ', 'PO8', 'PO4', 'POZ', 'O2', 'C6']     
chans_l = ['TP7', 'CP5', 'CP3', 'CP1', 'P7', 'P5', 'P3', 'P1', 'PO7', 'PO3', 'O1']
chans_par = ['PO4', 'P6']
chans_occ = ['OZ', 'POZ']
chans_v5 = ['TP8', 'CP6', 'CP4', 'CP2', 'P8', 'P6', 'P2'] #--> X
chans_v1 = ['PO8', 'PO4', 'OZ', 'POZ']


### Group 1 - Alpha-Alpha 0°
# signal1_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P02_Bsl_preprocessed_epo.fif')
# signal2_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P04_Bsl_preprocessed_epo.fif')
# signal3_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P06_Bsl_preprocessed_epo.fif')
# signal4_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P08_Bsl_preprocessed_epo.fif')
# signal5_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P09_Bsl_preprocessed_epo.fif')
# signal6_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P15_Bsl_preprocessed_epo.fif')
# signal7_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P17_Bsl_preprocessed_epo.fif')
# signal8_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P18_Bsl_preprocessed_epo.fif')
# signal9_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P21_Bsl_preprocessed_epo.fif')
# signal10_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P22_Bsl_preprocessed_epo.fif')
# signal11_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P23_Bsl_preprocessed_epo.fif')
# signal12_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P24_Bsl_preprocessed_epo.fif')
# signal13_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P44_Bsl_preprocessed_epo.fif')
# signal14_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P47_Bsl_preprocessed_epo.fif')
# signal15_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P48_Bsl_preprocessed_epo.fif')

# signal1_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P02_Post10_preprocessed_epo.fif')
# signal2_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P04_Post10_preprocessed_epo.fif')
# signal3_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P06_Post10_preprocessed_epo.fif')
# signal4_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P08_Post10_preprocessed_epo.fif')
# signal5_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P09_Post10_preprocessed_epo.fif')
# signal6_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P15_Post10_preprocessed_epo.fif')
# signal7_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P17_Post10_preprocessed_epo.fif')
# signal8_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P18_Post10_preprocessed_epo.fif')
# signal9_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P21_Post10_preprocessed_epo.fif')
# signal10_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P22_Post10_preprocessed_epo.fif')
# signal11_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P23_Post10_preprocessed_epo.fif')
# signal12_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P24_Post10_preprocessed_epo.fif')
# signal13_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P44_Post10_preprocessed_epo.fif')
# signal14_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P47_Post10_preprocessed_epo.fif')
# signal15_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P48_Post10_preprocessed_epo.fif')

# signal1_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P02_Post30_preprocessed_epo.fif')
# signal2_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P04_Post30_preprocessed_epo.fif')
# signal3_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P06_Post30_preprocessed_epo.fif')
# signal4_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P08_Post30_preprocessed_epo.fif')
# signal5_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P09_Post30_preprocessed_epo.fif')
# signal6_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P15_Post30_preprocessed_epo.fif')
# signal7_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P17_Post30_preprocessed_epo.fif')
# signal8_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P18_Post30_preprocessed_epo.fif')
# signal9_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P21_Post30_preprocessed_epo.fif')
# signal10_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P22_Post30_preprocessed_epo.fif')
# signal11_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P23_Post30_preprocessed_epo.fif')
# signal12_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P24_Post30_preprocessed_epo.fif')
# signal13_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P44_Post30_preprocessed_epo.fif')
# signal14_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P47_Post30_preprocessed_epo.fif')
# signal15_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P48_Post30_preprocessed_epo.fif')

# ### Group 2 - Alpha-Alpha 180°
signal1_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P01_Bsl_preprocessed_epo.fif')
signal2_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P05_Bsl_preprocessed_epo.fif')
signal3_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P11_Bsl_preprocessed_epo.fif')
signal4_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P14_Bsl_preprocessed_epo.fif')
signal5_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P20_Bsl_preprocessed_epo.fif')
signal6_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P26_Bsl_preprocessed_epo.fif')
signal7_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P27_Bsl_preprocessed_epo.fif')
signal8_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P28_Bsl_preprocessed_epo.fif')
signal9_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P29_Bsl_preprocessed_epo.fif')
signal10_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P30_Bsl_preprocessed_epo.fif')
signal11_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P32_Bsl_preprocessed_epo.fif')
signal12_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P34_Bsl_preprocessed_epo.fif')
signal13_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P35_Bsl_preprocessed_epo.fif')
signal14_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P40_Bsl_preprocessed_epo.fif')
signal15_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P43_Bsl_preprocessed_epo.fif')

signal1_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P01_Post10_preprocessed_epo.fif')
signal2_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P05_Post10_preprocessed_epo.fif')
signal3_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P11_Post10_preprocessed_epo.fif')
signal4_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P14_Post10_preprocessed_epo.fif')
signal5_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P20_Post10_preprocessed_epo.fif')
signal6_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P26_Post10_preprocessed_epo.fif')
signal7_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P27_Post10_preprocessed_epo.fif')
signal8_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P28_Post10_preprocessed_epo.fif')
signal9_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P29_Post10_preprocessed_epo.fif')
signal10_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P30_Post10_preprocessed_epo.fif')
signal11_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P32_Post10_preprocessed_epo.fif')
signal12_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P34_Post10_preprocessed_epo.fif')
signal13_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P35_Post10_preprocessed_epo.fif')
signal14_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P40_Post10_preprocessed_epo.fif')
signal15_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P43_Post10_preprocessed_epo.fif')

# signal1_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P01_Post30_preprocessed_epo.fif')
# signal2_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P05_Post30_preprocessed_epo.fif')
# signal3_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P11_Post30_preprocessed_epo.fif')
# signal4_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P14_Post30_preprocessed_epo.fif')
# signal5_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P20_Post30_preprocessed_epo.fif')
# signal6_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P26_Post30_preprocessed_epo.fif')
# signal7_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P27_Post30_preprocessed_epo.fif')
# signal8_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P28_Post30_preprocessed_epo.fif')
# signal9_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P29_Post30_preprocessed_epo.fif')
# signal10_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P30_Post30_preprocessed_epo.fif')
# signal11_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P32_Post30_preprocessed_epo.fif')
# signal12_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P34_Post30_preprocessed_epo.fif')
# signal13_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P35_Post30_preprocessed_epo.fif')
# signal14_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P40_Post30_preprocessed_epo.fif')
# signal15_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P43_Post30_preprocessed_epo.fif')

### Group 3 - AlphaV1-GammaV5
# signal1_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P10_Bsl_preprocessed_epo.fif')
# signal2_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P12_Bsl_preprocessed_epo.fif')
# signal3_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P16_Bsl_preprocessed_epo.fif')
# signal4_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P19_Bsl_preprocessed_epo.fif')
# signal5_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P33_Bsl_preprocessed_epo.fif')
# signal6_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P36_Bsl_preprocessed_epo.fif')
# signal7_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P37_Bsl_preprocessed_epo.fif')
# signal8_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P38_Bsl_preprocessed_epo.fif')
# signal9_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P39_Bsl_preprocessed_epo.fif')
# signal10_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P41_Bsl_preprocessed_epo.fif')
# signal11_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P42_Bsl_preprocessed_epo.fif')
# signal12_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P45_Bsl_preprocessed_epo.fif')
# signal13_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P46_Bsl_preprocessed_epo.fif')
# signal14_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P49_Bsl_preprocessed_epo.fif')
# signal15_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P50_Bsl_preprocessed_epo.fif')

# signal1_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P10_Post10_preprocessed_epo.fif')
# signal2_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P12_Post10_preprocessed_epo.fif')
# signal3_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P16_Post10_preprocessed_epo.fif')
# signal4_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P19_Post10_preprocessed_epo.fif')
# signal5_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P33_Post10_preprocessed_epo.fif')
# signal6_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P36_Post10_preprocessed_epo.fif')
# signal7_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P37_Post10_preprocessed_epo.fif')
# signal8_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P38_Post10_preprocessed_epo.fif')
# signal9_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P39_Post10_preprocessed_epo.fif')
# signal10_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P41_Post10_preprocessed_epo.fif')
# signal11_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P42_Post10_preprocessed_epo.fif')
# signal12_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P45_Post10_preprocessed_epo.fif')
# signal13_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P46_Post10_preprocessed_epo.fif')
# signal14_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P49_Post10_preprocessed_epo.fif')
# signal15_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P50_Post10_preprocessed_epo.fif')

# signal1_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P10_Post30_preprocessed_epo.fif')
# signal2_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P12_Post30_preprocessed_epo.fif')
# signal3_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P16_Post30_preprocessed_epo.fif')
# signal4_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P19_Post30_preprocessed_epo.fif')
# signal5_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P33_Post30_preprocessed_epo.fif')
# signal6_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P36_Post30_preprocessed_epo.fif')
# signal7_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P37_Post30_preprocessed_epo.fif')
# signal8_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P38_Post30_preprocessed_epo.fif')
# signal9_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P39_Post30_preprocessed_epo.fif')
# signal10_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P41_Post30_preprocessed_epo.fif')
# signal11_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P42_Post30_preprocessed_epo.fif')
# signal12_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P45_Post30_preprocessed_epo.fif')
# signal13_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P46_Post30_preprocessed_epo.fif')
# signal14_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P49_Post30_preprocessed_epo.fif')
# signal15_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P50_Post30_preprocessed_epo.fif')

### Group 4 - GammaV1-AlphaV5
# signal1_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P51_Bsl_preprocessed_epo.fif')
# signal2_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P52_Bsl_preprocessed_epo.fif')
# signal3_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P53_Bsl_preprocessed_epo.fif')
# signal4_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P54_Bsl_preprocessed_epo.fif')
# signal5_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P55_Bsl_preprocessed_epo.fif')
# signal6_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P56_Bsl_preprocessed_epo.fif')
# signal7_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P57_Bsl_preprocessed_epo.fif')
# signal8_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P58_Bsl_preprocessed_epo.fif')
# signal9_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P60_Bsl_preprocessed_epo.fif')
# signal10_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P61_Bsl_preprocessed_epo.fif')
# signal11_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P65_Bsl_preprocessed_epo.fif')
# signal12_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P66_Bsl_preprocessed_epo.fif')
# signal13_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P67_Bsl_preprocessed_epo.fif')
# signal14_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P68_Bsl_preprocessed_epo.fif')
# signal15_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P70_Bsl_preprocessed_epo.fif')

# signal1_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P51_Post10_preprocessed_epo.fif')
# signal2_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P52_Post10_preprocessed_epo.fif')
# signal3_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P53_Post10_preprocessed_epo.fif')
# signal4_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P54_Post10_preprocessed_epo.fif')
# signal5_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P55_Post10_preprocessed_epo.fif')
# signal6_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P56_Post10_preprocessed_epo.fif')
# signal7_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P57_Post10_preprocessed_epo.fif')
# signal8_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P58_Post10_preprocessed_epo.fif')
# signal9_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P60_Post10_preprocessed_epo.fif')
# signal10_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P61_Post10_preprocessed_epo.fif')
# signal11_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P65_Post10_preprocessed_epo.fif')
# signal12_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P66_Post10_preprocessed_epo.fif')
# signal13_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P67_Post10_preprocessed_epo.fif')
# signal14_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P68_Post10_preprocessed_epo.fif')
# signal15_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P70_Post10_preprocessed_epo.fif')

# signal1_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P51_Post30_preprocessed_epo.fif')
# signal2_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P52_Post30_preprocessed_epo.fif')
# signal3_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P53_Post30_preprocessed_epo.fif')
# signal4_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P54_Post30_preprocessed_epo.fif')
# signal5_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P55_Post30_preprocessed_epo.fif')
# signal6_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P56_Post30_preprocessed_epo.fif')
# signal7_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P57_Post30_preprocessed_epo.fif')
# signal8_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P58_Post30_preprocessed_epo.fif')
# signal9_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P60_Post30_preprocessed_epo.fif')
# signal10_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P61_Post30_preprocessed_epo.fif')
# signal11_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P65_Post30_preprocessed_epo.fif')
# signal12_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P66_Post30_preprocessed_epo.fif')
# signal13_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P67_Post30_preprocessed_epo.fif')
# signal14_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P68_Post30_preprocessed_epo.fif')
# signal15_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P70_Post30_preprocessed_epo.fif')

### Group 5 - SHAM
# signal1_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P59_Bsl_preprocessed_epo.fif')
# signal2_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P62_Bsl_preprocessed_epo.fif')
# signal3_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P63_Bsl_preprocessed_epo.fif')
# signal4_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P64_Bsl_preprocessed_epo.fif')
# signal5_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P69_Bsl_preprocessed_epo.fif')
# signal6_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P71_Bsl_preprocessed_epo.fif')
# signal7_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P72_Bsl_preprocessed_epo.fif')
# signal8_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P73_Bsl_preprocessed_epo.fif')
# signal9_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P74_Bsl_preprocessed_epo.fif')
# signal10_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P75_Bsl_preprocessed_epo.fif')
# signal11_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P76_Bsl_preprocessed_epo.fif')
# signal12_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P77_Bsl_preprocessed_epo.fif')
# signal13_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P78_Bsl_preprocessed_epo.fif')
# signal14_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P80_Bsl_preprocessed_epo.fif')
# signal15_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P81_Bsl_preprocessed_epo.fif')

# signal1_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P59_Post10_preprocessed_epo.fif')
# signal2_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P62_Post10_preprocessed_epo.fif')
# signal3_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P63_Post10_preprocessed_epo.fif')
# signal4_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P64_Post10_preprocessed_epo.fif')
# signal5_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P69_Post10_preprocessed_epo.fif')
# signal6_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P71_Post10_preprocessed_epo.fif')
# signal7_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P72_Post10_preprocessed_epo.fif')
# signal8_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P73_Post10_preprocessed_epo.fif')
# signal9_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P74_Post10_preprocessed_epo.fif')
# signal10_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P75_Post10_preprocessed_epo.fif')
# signal11_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P76_Post10_preprocessed_epo.fif')
# signal12_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P77_Post10_preprocessed_epo.fif')
# signal13_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P78_Post10_preprocessed_epo.fif')
# signal14_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P80_Post10_preprocessed_epo.fif')
# signal15_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P81_Post10_preprocessed_epo.fif')

# signal1_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P59_Post30_preprocessed_epo.fif')
# signal2_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P62_Post30_preprocessed_epo.fif')
# signal3_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P63_Post30_preprocessed_epo.fif')
# signal4_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P64_Post30_preprocessed_epo.fif')
# signal5_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P69_Post30_preprocessed_epo.fif')
# signal6_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P71_Post30_preprocessed_epo.fif')
# signal7_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P72_Post30_preprocessed_epo.fif')
# signal8_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P73_Post30_preprocessed_epo.fif')
# signal9_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P74_Post30_preprocessed_epo.fif')
# signal10_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P75_Post30_preprocessed_epo.fif')
# signal11_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P76_Post30_preprocessed_epo.fif')
# signal12_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P77_Post30_preprocessed_epo.fif')
# signal13_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P78_Post30_preprocessed_epo.fif')
# signal14_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P80_Post30_preprocessed_epo.fif')
# signal15_P3 = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/1st_2nd_Experiment/P81_Post30_preprocessed_epo.fif')


G_Bsl = (signal1_Bsl, signal2_Bsl, signal3_Bsl, signal4_Bsl, signal5_Bsl, 
      signal6_Bsl, signal7_Bsl, signal8_Bsl, signal9_Bsl, signal10_Bsl, 
      signal11_Bsl, signal12_Bsl, signal13_Bsl, signal14_Bsl, signal15_Bsl)

G_P = (signal1_P, signal2_P, signal3_P, signal4_P, signal5_P,
     signal6_P, signal7_P, signal8_P, signal9_P, signal10_P, 
     signal11_P, signal12_P, signal13_P, signal14_P, signal15_P)

# G_P3 = (signal1_P3, signal2_P3, signal3_P3, signal4_P3, signal5_P3, 
#         signal6_P3, signal7_P3, signal8_P3, signal9_P3, signal10_P3, 
#         signal11_P3, signal12_P3, signal13_P3, signal14_P3, signal15_P3)



# Group_TF(G_Bsl, G_P3, 'stim_cor', 'stim_cor', chans_r, [0], 'G4_P3-Bsl_r', 'power')
# Group_TF(G_Bsl, G_P3, 'stim_cor', 'stim_cor', chans_r, 0, 'G1_P3-Bsl_r', 'power')
# Group_TF(G_Bsl, G_P, 'stim_cor', 'stim_cor', chans_l, 'G4_P1-Bsl_l', 'power') #Control Condition Left Hemisphere

# Group_TF(G_Bsl, G_P, 'stim_cor', 'stim_cor', chans_par, chans_occ, 'G5_P1-Bsl', 'phase')
# Group_TF(G_Bsl, G_P3, 'stim_cor', 'stim_cor', chans_par, chans_occ, 'G4_P3-Bsl_occ', 'phase')

# Group_Gr(G_Bsl, G_P, 'stim_cor', 'stim_cor', chans_v5, chans_v1, 'G4_P1-Bsl')
# Group_Gr(G_Bsl, G_P3, 'stim_cor', 'stim_cor', chans_v5, chans_v1, 'G4_P3-Bsl')

# Group_Src(G_Bsl, G_P3, 'stim_cor', 'stim_cor', 'G1_Bsl', 'G1_P30', 'psd_v1v5')
# Group_Src(G_Bsl, G_P, 'stim_cor', 'stim_cor', 'G5_Bsl', 'G5_P10', 'tf_hilb')
# Group_Src(G_Bsl, G_P, 'stim_cor', 'stim_cor', 'G5_Bsl', 'G5_P10', 'tf_power')
Group_Src(G_Bsl, G_P, 'stim_cor', 'stim_cor', 'G2_Bsl', 'G2_P10', 'connect')
# Group_Src(G_Bsl, G_P, 'stim_cor', 'stim_cor', 'G3_P30', 'G4_P30', 'cross_freq')
# Group_Src(G_Bsl, G_P3, 'stim_cor', 'stim_cor', 'G5_Bsl', 'G5_P30', 'cross_corr')

# Group_TF(G_Bsl, G_P, 'stim_cor', 'stim_cor', chans_rh, chans_rh, 'rh_G5', 'connect_all')
# Group_TF(G_Bsl, G_P, 'stim_cor', 'stim_cor', chans_par, chans_par, 'par_G5', 'pow_spec_den') #All_G5_P3-Bsl
# Group_TF(G_Bsl, G_P, 'stim_cor', 'stim_cor', chans_all, chans_all, 'G4_pwrAll', 'brain_topo_tf')
