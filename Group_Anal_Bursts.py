import numpy as np
import scipy
import mne
import matplotlib
import matplotlib.pyplot as plt

from TimeFreqAna import *
from Group_TF import *

# from GrangerCaus import *
# from Group_Gr import *

from SourcesReco_Bursts import *
from Group_Src_Bursts import *

### PILOTS
# signal1_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/Es_preprocessed_epo.fif')
# signal2_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/Jul_preprocessed_epo.fif')
# signal3_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/Mih_preprocessed_epo.fif')
# signal4_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/Fab_preprocessed_epo.fif')
# signal5_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/Mic_preprocessed_epo.fif')
# signal6_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/Pie_preprocessed_epo.fif')
# signal7_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/B_Gas_preprocessed_epo.fif')
# signal8_Bsl = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/B_Fl2_preprocessed_epo.fif')

### Group 1 - Alpha-Alpha 0°
asignal1_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P03_Bursts_preprocessed_epo.fif')
asignal2_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P05_Bursts_preprocessed_epo.fif')
asignal3_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P06_Bursts_preprocessed_epo.fif')
asignal4_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P07_Bursts_preprocessed_epo.fif')
asignal5_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P10_Bursts_preprocessed_epo.fif')
asignal6_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P12_Bursts_preprocessed_epo.fif')
asignal7_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P13_Bursts_preprocessed_epo.fif')
asignal8_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P17_Bursts_preprocessed_epo.fif')
asignal9_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P18_Bursts_preprocessed_epo.fif')
asignal10_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P19_Bursts_preprocessed_epo.fif')
asignal11_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P21_Bursts_preprocessed_epo.fif')
asignal12_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P23_Bursts_preprocessed_epo.fif')
asignal13_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P25_Bursts_preprocessed_epo.fif')
asignal14_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P26_Bursts_preprocessed_epo.fif')
asignal15_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P29_Bursts_preprocessed_epo.fif')

### Group 2 - Alpha-Alpha 180°
signal1_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P01_Bursts_preprocessed_epo.fif')
signal2_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P02_Bursts_preprocessed_epo.fif')
signal3_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P04_Bursts_preprocessed_epo.fif')
signal4_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P08_Bursts_preprocessed_epo.fif')
signal5_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P09_Bursts_preprocessed_epo.fif')
signal6_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P11_Bursts_preprocessed_epo.fif')
signal7_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P14_Bursts_preprocessed_epo.fif')
signal8_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P15_Bursts_preprocessed_epo.fif')
signal9_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P16_Bursts_preprocessed_epo.fif')
signal10_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P20_Bursts_preprocessed_epo.fif')
signal11_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P22_Bursts_preprocessed_epo.fif')
signal12_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P24_Bursts_preprocessed_epo.fif')
signal13_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P27_Bursts_preprocessed_epo.fif')
signal14_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P28_Bursts_preprocessed_epo.fif')
signal15_P = mne.read_epochs('/home/robertofelipe_sg/Documents/Preprocessed/Bursts/P30_Bursts_preprocessed_epo.fif')


G_P = (asignal1_P, asignal2_P, asignal3_P, asignal4_P, asignal5_P,
     asignal6_P, asignal7_P, asignal8_P, asignal9_P, asignal10_P, 
     asignal11_P, asignal12_P, asignal13_P, asignal14_P, asignal15_P)

# G_P = (signal1_P, signal2_P, signal3_P, signal4_P, signal5_P,
#      signal6_P, signal7_P, signal8_P, signal9_P, signal10_P, 
#      signal11_P, signal12_P, signal13_P, signal14_P, signal15_P)

aG_P = (signal1_P, signal2_P, signal3_P, signal4_P, signal5_P,
     signal6_P, signal7_P, signal8_P, signal9_P, signal10_P, 
     signal11_P, signal12_P, signal13_P, signal14_P, signal15_P,
       asignal1_P, asignal2_P, asignal3_P, asignal4_P, asignal5_P,
     asignal6_P, asignal7_P, asignal8_P, asignal9_P, asignal10_P, 
     asignal11_P, asignal12_P, asignal13_P, asignal14_P, asignal15_P)


# Group_TF(G_Bsl, G_P3, 'stim_cor', 'stim_cor', chans_r, [0], 'G4_P3-Bsl_r', 'power')
# Group_TF(G_Bsl, G_P3, 'stim_cor', 'stim_cor', chans_r, 0, 'G1_P3-Bsl_r', 'power')
# Group_TF(G_Bsl, G_P, 'stim_cor', 'stim_cor', chans_l, 'G4_P1-Bsl_l', 'power') #Control Condition Left Hemisphere

# Group_TF(G_Bsl, G_P, 'stim_cor', 'stim_cor', chans_par, chans_occ, 'G5_P1-Bsl', 'phase')
# Group_TF(G_Bsl, G_P3, 'stim_cor', 'stim_cor', chans_par, chans_occ, 'G4_P3-Bsl_occ', 'phase')

# Group_Gr(G_Bsl, G_P, 'stim_cor', 'stim_cor', chans_v5, chans_v1, 'G4_P1-Bsl')
# Group_Gr(G_Bsl, G_P3, 'stim_cor', 'stim_cor', chans_v5, chans_v1, 'G4_P3-Bsl')

# Group_Src(G_Bsl, G_P3, 'stim_cor', 'stim_cor', 'G1_Bsl', 'G1_P30', 'psd_v1v5')
# Group_Src(G_Bsl, G_P, 'stim_cor', 'stim_cor', 'G5_Bsl', 'G5_P10', 'tf_hilb')
Group_Src_Bursts(G_P, G_P, 'verum_cor', 'sham_cor', 'G1_Bursts', 'G1_BSham', 'tf_wave')
# Group_Src_Bursts(G_P, G_P, 'verum_cor', 'sham_cor', 'G2_Bursts', 'G2_BSham', 'tf_power')
# Group_Src_Bursts(G_P, G_P, 'verum_cor', 'sham_cor', 'G1_Bursts', 'G1_BSham', 'connect')
# Group_Src_Bursts(G_P, aG_P, 'verum_cor', 'sham_cor', 'G1', 'G2', 'cross_freq')
# Group_Src(G_Bsl, G_P3, 'stim_cor', 'stim_cor', 'G5_Bsl', 'G5_P30', 'cross_corr')

# Group_TF(G_Bsl, G_P, 'stim_cor', 'stim_cor', chans_rh, chans_rh, 'rh_G5', 'connect_all')
# Group_TF(G_Bsl, G_P, 'stim_cor', 'stim_cor', chans_par, chans_par, 'par_G5', 'pow_spec_den') #All_G5_P3-Bsl
# Group_TF(G_Bsl, G_P, 'stim_cor', 'stim_cor', chans_all, chans_all, 'G4_pwrAll', 'brain_topo_tf')
