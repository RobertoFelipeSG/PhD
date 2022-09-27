import numpy as np
import pandas as pd
import scipy
import mne
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
#matplotlib.use('TkAgg')

from Stats_TF import *
from Stats_Sigs import *
from Stats_Univar import *
from Stats_CF import *

##### Power
# P10
# G1_Pre = np.load('../../../Documents/09.19/Power/TimeFreq/Pre_G1_P1-Bsl_r.npy') # sub x frex x time-points
# G1_P10 = np.load('../../../Documents/09.19/Power/TimeFreq/Pos_G1_P1-Bsl_r.npy')
# G2_Pre = np.load('../../../Documents/09.19/Power/TimeFreq/Pre_G2_P1-Bsl_r.npy') # sub x frex x time-points
# G2_P10 = np.load('../../../Documents/09.19/Power/TimeFreq/Pos_G2_P1-Bsl_r.npy')
# G3_Pre = np.load('../../../Documents/09.19/Power/TimeFreq/Pre_G3_P1-Bsl_r.npy') # sub x frex x time-points
# G3_P10 = np.load('../../../Documents/09.19/Power/TimeFreq/Pos_G3_P1-Bsl_r.npy')
# G4_Pre = np.load('../../../Documents/09.19/Power/TimeFreq/Pre_G4_P1-Bsl_r.npy') # sub x frex x time-points
# G4_P10 = np.load('../../../Documents/09.19/Power/TimeFreq/Pos_G4_P1-Bsl_r.npy')
# G5_Pre = np.load('../../../Documents/09.19/Power/TimeFreq/Pre_G5_P1-Bsl_r.npy') # sub x frex x time-points
# G5_P10 = np.load('../../../Documents/09.19/Power/TimeFreq/Pos_G5_P1-Bsl_r.npy')

# Stats_TF(G1_Pre, G1_P10, 'G1_P1-Bsl', 'TF')
# Stats_TF(G2_Pre, G2_P10, 'G2_P1-Bsl', 'TF')
# Stats_TF(G3_Pre, G3_P10, 'G3_P1-Bsl', 'TF')
# Stats_TF(G4_Pre, G4_P10, 'G4_P1-Bsl', 'TF')
# Stats_TF(G5_Pre, G5_P10, 'G5_P1-Bsl', 'TF')

# P30
# G1_Pre = np.load('../../../Documents/09.19/Power/TimeFreq/Pre_G1_P3-Bsl_r.npy') # sub x frex x time-points
# G1_P30 = np.load('../../../Documents/09.19/Power/TimeFreq/Pos_G1_P3-Bsl_r.npy')
# G2_Pre = np.load('../../../Documents/09.19/Power/TimeFreq/Pre_G2_P3-Bsl_r.npy') # sub x frex x time-points
# G2_P30 = np.load('../../../Documents/09.19/Power/TimeFreq/Pos_G2_P3-Bsl_r.npy')
# G3_Pre = np.load('../../../Documents/09.19/Power/TimeFreq/Pre_G3_P3-Bsl_r.npy') # sub x frex x time-points
# G3_P30 = np.load('../../../Documents/09.19/Power/TimeFreq/Pos_G3_P3-Bsl_r.npy')
# G4_Pre = np.load('../../../Documents/09.19/Power/TimeFreq/Pre_G4_P3-Bsl_r.npy') # sub x frex x time-points
# G4_P30 = np.load('../../../Documents/09.19/Power/TimeFreq/Pos_G4_P3-Bsl_r.npy')
# G5_Pre = np.load('../../../Documents/09.19/Power/TimeFreq/Pre_G5_P3-Bsl_r.npy') # sub x frex x time-points
# G5_P30 = np.load('../../../Documents/09.19/Power/TimeFreq/Pos_G5_P3-Bsl_r.npy')

# Stats_TF(G1_Pre, G1_P30, 'G1_P3-Bsl', 'TF')
# Stats_TF(G2_Pre, G2_P30, 'G2_P3-Bsl', 'TF')
# Stats_TF(G3_Pre, G3_P30, 'G3_P3-Bsl', 'TF')
# Stats_TF(G4_Pre, G4_P30, 'G4_P3-Bsl', 'TF')
# Stats_TF(G5_Pre, G5_P30, 'G5_P3-Bsl', 'TF')


## Between groups

# G1_Diff = G1_Pos - G1_Pre
# G2_Diff = G2_Pos - G2_Pre
# G3_Diff = G3_Pos - G3_Pre
# G4_Diff = G4_Pos - G4_Pre

# Stats_TF(G1_Diff, G2_Diff, 'G1G2_P1-Bsl', 'TF')
# Stats_TF(G3_Diff, G4_Diff, 'G3G4_P1-Bsl', 'TF')

## Different groups + same condition

# Stats_TF(G1_Pre, G2_Pre, 'G1G2_Bsl', 'TF')
# Stats_TF(G1_Pos, G2_Pos, 'G1G2_P1', 'TF')
# Stats_TF(G3_Pre, G4_Pre, 'G3G4_Bsl', 'TF')
# Stats_TF(G3_Pos, G4_Pos, 'G3G4_P1', 'TF')


## Correlation with Behavior

# Behavs = pd.read_excel('Behav_data.xlsx')

# ##### ACTIVE GROUPS

# Groups = (G2_Pre, G1_Pre, G3_Pre, G4_Pre, G2_Pos, G1_Pos, G3_Pos, G4_Pos)

# maxx = np.empty([15,4,8]) # Subs. x Freqs.Bands x Groups(cond)
# avgx = np.empty([15,4,8])

# for g, gr in enumerate(Groups):
#     for sub in range(gr.shape[0]):
#         maxx[sub,0,g] = np.max(np.max(gr[sub,0:5,375:500],1),0)
#         maxx[sub,1,g] = np.max(np.max(gr[sub,5:12,375:500],1),0)
#         maxx[sub,2,g] = np.max(np.max(gr[sub,12:29,375:500],1),0)
#         maxx[sub,3,g] = np.max(np.max(gr[sub,29:,375:500],1),0)
#         avgx[sub,0,g] = np.mean(np.mean(gr[sub,0:5,375:500],1),0)
#         avgx[sub,1,g] = np.mean(np.mean(gr[sub,5:12,375:500],1),0)
#         avgx[sub,2,g] = np.mean(np.mean(gr[sub,12:29,375:500],1),0)
#         avgx[sub,3,g] = np.mean(np.mean(gr[sub,29:40,375:500],1),0)
        
# # To a PD DataFrame      

# label_Pre_max = np.array(['The_Pre_max', 'Alp_Pre_max', 'Bet_Pre_max', 'Gam_Pre_max'])
# label_Pos_max = np.array(['The_Pos_max', 'Alp_Pos_max', 'Bet_Pos_max', 'Gam_Pos_max'])
# Eeg_Pre_max = pd.DataFrame(data=(np.transpose(maxx[:,:,0:4],(0,2,1))).reshape((60,4)), columns=label_Pre_max)
# Eeg_Pos_max = pd.DataFrame(data=(np.transpose(maxx[:,:,4:8],(0,2,1))).reshape((60,4)), columns=label_Pos_max)

# label_Pre_avg = np.array(['The_Pre_avg', 'Alp_Pre_avg', 'Bet_Pre_avg', 'Gam_Pre_avg'])
# label_Pos_avg = np.array(['The_Pos_avg', 'Alp_Pos_avg', 'Bet_Pos_avg', 'Gam_Pos_avg'])
# Eeg_Pre_avg = pd.DataFrame(data=(np.transpose(avgx[:,:,0:4],(0,2,1))).reshape((60,4)), columns=label_Pre_avg)
# Eeg_Pos_avg = pd.DataFrame(data=(np.transpose(avgx[:,:,4:8],(0,2,1))).reshape((60,4)), columns=label_Pos_avg)

# combined = pd.concat([Behavs,Eeg_Pre_max,Eeg_Pos_max,Eeg_Pre_avg,Eeg_Pos_avg], axis=1)
# print(combined)
# combined.to_csv('Max_and_Avg.txt', sep='\t')

# # Plotting

#df = pd.plotting.scatter_matrix(combined[['Baseline','Post10','The_Pre','Alp_Pre','Bet_Pre','The_Pos','Alp_Pos','Bet_Pos']])

# cor_mat = combined['Baseline'].corr(combined['Theta_Pre'])
# print(type(cor_mat))
#pd.plotting.scatter_matrix(combined[['Baseline'[0:15],'Post10'[0:15],'The_Pre'[0:15],'Alp_Pre'[0:15],'Bet_Pre'[0:15],'The_Pos'[0:15],'Alp_Pos'[0:15],'Bet_Pos'[0:15]]])
#pd.plotting.scatter_matrix(combined[['Baseline'[15:30],'Post10'[15:30],'The_Pre'[15:30],'Alp_Pre'[15:30],'Bet_Pre'[15:30],'The_Pos'[15:30],'Alp_Pos'[15:30],'Bet_Pos'[15:30]]])
#pd.plotting.scatter_matrix(combined[['Baseline'[0:15],'Post10'[30:45],'The_Pre'[30:45],'Alp_Pre'[30:45],'Bet_Pre'[30:45],'The_Pos'[30:45],'Alp_Pos'[30:45],'Bet_Pos'[30:45]]])
#pd.plotting.scatter_matrix(combined[['Baseline'[45:60],'Post10'[45:60],'The_Pre'[45:60],'Alp_Pre'[45:60],'Bet_Pre'[45:60],'The_Pos'[45:60],'Alp_Pos'[45:60],'Bet_Pos'[45:60]]])

#plt.show()

# plt.matshow(combined[['Baseline','Post10','The_Pre','Alp_Pre','Bet_Pre','The_Pos','Alp_Pos','Bet_Pos']])
# plt.xticks(range(8), ['Baseline','Post10','The_Pre','Alp_Pre','Bet_Pre','The_Pos','Alp_Pos','Bet_Pos'])
# plt.yticks(range(8), ['Baseline','Post10','The_Pre','Alp_Pre','Bet_Pre','The_Pos','Alp_Pos','Bet_Pos'])
# plt.colorbar()
# plt.show()

###### SHAM GROUP

# GroupsSh = (G5_Pre, G5_Pos)
# maxxSh = np.empty([8,4,2]) # Subs. x Freqs.Bands x Groups(cond)
# avgxSh = np.empty([8,4,2])

# for gSh, grSh in enumerate(GroupsSh):
#     for subSh in range(grSh.shape[0]):
#         maxxSh[subSh,0,gSh] = np.max(np.max(grSh[subSh,0:4,375:500],1),0)
#         maxxSh[subSh,1,gSh] = np.max(np.max(grSh[subSh,5:11,375:500],1),0)
#         maxxSh[subSh,2,gSh] = np.max(np.max(grSh[subSh,12:27,375:500],1),0)
#         maxxSh[subSh,3,gSh] = np.max(np.max(grSh[subSh,28:,375:500],1),0)
#         avgxSh[subSh,0,gSh] = np.mean(np.mean(grSh[subSh,0:4,375:500],1),0)
#         avgxSh[subSh,1,gSh] = np.mean(np.mean(grSh[subSh,5:11,375:500],1),0)
#         avgxSh[subSh,2,gSh] = np.mean(np.mean(grSh[subSh,12:27,375:500],1),0)
#         avgxSh[subSh,3,gSh] = np.mean(np.mean(grSh[subSh,28:,375:500],1),0)
        
# # To a PD DataFrame   
        
# label_Pre_maxSh = np.array(['The_Pre_max', 'Alp_Pre_max', 'Bet_Pre_max', 'Gam_Pre_max'])
# label_Pos_maxSh = np.array(['The_Pos_max', 'Alp_Pos_max', 'Bet_Pos_max', 'Gam_Pos_max'])
# Eeg_Pre_maxSh = pd.DataFrame(data=(np.transpose(maxxSh[:,:,:1],(0,2,1))).reshape((-1,4)), columns=label_Pre_maxSh)
# Eeg_Pos_maxSh = pd.DataFrame(data=(np.transpose(maxxSh[:,:,1:],(0,2,1))).reshape((-1,4)), columns=label_Pos_maxSh)

# label_Pre_avgSh = np.array(['The_Pre_avg', 'Alp_Pre_avg', 'Bet_Pre_avg', 'Gam_Pre_avg'])
# label_Pos_avgSh = np.array(['The_Pos_avg', 'Alp_Pos_avg', 'Bet_Pos_avg', 'Gam_Pos_avg'])
# Eeg_Pre_avgSh = pd.DataFrame(data=(np.transpose(avgxSh[:,:,:1],(0,2,1))).reshape((-1,4)), columns=label_Pre_avgSh)
# Eeg_Pos_avgSh = pd.DataFrame(data=(np.transpose(avgxSh[:,:,1:],(0,2,1))).reshape((-1,4)), columns=label_Pos_avgSh)
# combinedSh = pd.concat([Behavs,Eeg_Pre_maxSh,Eeg_Pos_maxSh,Eeg_Pre_avgSh,Eeg_Pos_avgSh], axis=1)
# print(combinedSh)
# combinedSh.to_csv('Max_and_Avg_Sham.txt', sep='\t')


        

### Connectivity

# Analysis over Time
# G1_Pre_t = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G1/PLV/Pre_t_G1_P1-Bsl.npy'),1))
# G1_Pos_t = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G1/PLV/Pos_t_G1_P1-Bsl.npy'),1))
# G1_Pre_t_avg = np.load('../../../Documents/Electrodes_May19/G1/PLV/Pre_t_G1_P1-Bsl_avg.npy')
# G1_Pos_t_avg = np.load('../../../Documents/Electrodes_May19/G1/PLV/Pos_t_G1_P1-Bsl_avg.npy')

# G2_Pre_t = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G2/PLV/Pre_t_G2_P1-Bsl.npy'),1))
# G2_Pos_t = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G2/PLV/Pos_t_G2_P1-Bsl.npy'),1))
# G2_Pre_t_avg = np.load('../../../Documents/Electrodes_May19/G2/PLV/Pre_t_G2_P1-Bsl_avg.npy')
# G2_Pos_t_avg = np.load('../../../Documents/Electrodes_May19/G2/PLV/Pos_t_G2_P1-Bsl_avg.npy')

# G3_Pre_t = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G3/PLV/Pre_t_G3_P1-Bsl.npy'),1))
# G3_Pos_t = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G3/PLV/Pos_t_G3_P1-Bsl.npy'),1))
# G3_Pre_t_avg = np.load('../../../Documents/Electrodes_May19/G3/PLV/Pre_t_G3_P1-Bsl_avg.npy')
# G3_Pos_t_avg = np.load('../../../Documents/Electrodes_May19/G3/PLV/Pos_t_G3_P1-Bsl_avg.npy')

# G4_Pre_t = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G4/PLV/Pre_t_G4_P1-Bsl.npy'),1))
# G4_Pos_t = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G4/PLV/Pos_t_G4_P1-Bsl.npy'),1))
# G4_Pre_t_avg = np.load('../../../Documents/Electrodes_May19/G4/PLV/Pre_t_G4_P1-Bsl_avg.npy')
# G4_Pos_t_avg = np.load('../../../Documents/Electrodes_May19/G4/PLV/Pos_t_G4_P1-Bsl_avg.npy')

# G5_Pre_t = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G5/PLV/Pre_t_G5_P1-Bsl.npy'),1))
# G5_Pos_t = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G5/PLV/Pos_t_G5_P1-Bsl.npy'),1))
# G5_Pre_t_avg = np.load('../../../Documents/Electrodes_May19/G5/PLV/Pre_t_G5_P1-Bsl_avg.npy')
# G5_Pos_t_avg = np.load('../../../Documents/Electrodes_May19/G5/PLV/Pos_t_G5_P1-Bsl_avg.npy')

# Stats_Univar(G1_Pre_t, G1_Pos_t, G1_Pre_t_avg, G1_Pos_t_avg, 't_G1_P1-Bsl')
# Stats_Univar(G2_Pre_t, G2_Pos_t, G2_Pre_t_avg, G2_Pos_t_avg, 't_G2_P1-Bsl')
# Stats_Univar(G3_Pre_t, G3_Pos_t, G3_Pre_t_avg, G3_Pos_t_avg, 't_G3_P1-Bsl')
# Stats_Univar(G4_Pre_t, G4_Pos_t, G4_Pre_t_avg, G4_Pos_t_avg, 't_G4_P1-Bsl')
# Stats_Univar(G5_Pre_t, G5_Pos_t, G5_Pre_t_avg, G5_Pos_t_avg, 't_G5_P1-Bsl')

# # Analysis over Frequency
# G1_Pre_f = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G1/PLV/Pre_t_G1_P1-Bsl.npy'),2))
# G1_Pos_f = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G1/PLV/Pos_t_G1_P1-Bsl.npy'),2))
# G1_Pre_f_avg = np.load('../../../Documents/Electrodes_May19/G1/PLV/Pre_f_G1_P1-Bsl_avg.npy')
# G1_Pos_f_avg = np.load('../../../Documents/Electrodes_May19/G1/PLV/Pos_f_G1_P1-Bsl_avg.npy')

# G2_Pre_f = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G2/PLV/Pre_t_G2_P1-Bsl.npy'),2))
# G2_Pos_f = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G2/PLV/Pos_t_G2_P1-Bsl.npy'),2))
# G2_Pre_f_avg = np.load('../../../Documents/Electrodes_May19/G2/PLV/Pre_f_G2_P1-Bsl_avg.npy')
# G2_Pos_f_avg = np.load('../../../Documents/Electrodes_May19/G2/PLV/Pos_f_G2_P1-Bsl_avg.npy')

# G3_Pre_f = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G3/PLV/Pre_t_G3_P1-Bsl.npy'),2))
# G3_Pos_f = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G3/PLV/Pos_t_G3_P1-Bsl.npy'),2))
# G3_Pre_f_avg = np.load('../../../Documents/Electrodes_May19/G3/PLV/Pre_f_G3_P1-Bsl_avg.npy')
# G3_Pos_f_avg = np.load('../../../Documents/Electrodes_May19/G3/PLV/Pos_f_G3_P1-Bsl_avg.npy')

# G4_Pre_f = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G4/PLV/Pre_t_G4_P1-Bsl.npy'),2))
# G4_Pos_f = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G4/PLV/Pos_t_G4_P1-Bsl.npy'),2))
# G4_Pre_f_avg = np.load('../../../Documents/Electrodes_May19/G4/PLV/Pre_f_G4_P1-Bsl_avg.npy')
# G4_Pos_f_avg = np.load('../../../Documents/Electrodes_May19/G4/PLV/Pos_f_G4_P1-Bsl_avg.npy')

# G5_Pre_f = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G5/PLV/Pre_t_G5_P1-Bsl.npy'),2))
# G5_Pos_f = abs(np.mean(np.load('../../../Documents/Electrodes_May19/G5/PLV/Pos_t_G5_P1-Bsl.npy'),2))
# G5_Pre_f_avg = np.load('../../../Documents/Electrodes_May19/G5/PLV/Pre_f_G5_P1-Bsl_avg.npy')
# G5_Pos_f_avg = np.load('../../../Documents/Electrodes_May19/G5/PLV/Pos_f_G5_P1-Bsl_avg.npy')

# Stats_Univar(G1_Pre_f, G1_Pos_f, G1_Pre_f_avg, G1_Pos_f_avg, 't_G1_P1-Bsl')
# Stats_Univar(G2_Pre_f, G2_Pos_f, G2_Pre_f_avg, G2_Pos_f_avg, 't_G2_P1-Bsl')
# Stats_Univar(G3_Pre_f, G3_Pos_f, G3_Pre_f_avg, G3_Pos_f_avg, 't_G3_P1-Bsl')
# Stats_Univar(G4_Pre_f, G4_Pos_f, G4_Pre_f_avg, G4_Pos_f_avg, 't_G4_P1-Bsl')
# Stats_Univar(G5_Pre_f, G5_Pos_f, G5_Pre_f_avg, G5_Pos_f_avg, 't_G5_P1-Bsl')


### G-Causality

# Time
# # # Pre
# G1_Pre_x2y = np.mean(np.squeeze(np.load('G1-4/Pre_x2y_G1_P1-Bsl.npy')),0)
# G1_Pre_y2x = np.mean(np.squeeze(np.load('G1-4/Pre_y2x_G1_P1-Bsl.npy')),0)

# G2_Pre_x2y = np.mean(np.squeeze(np.load('G2-4/Pre_x2y_G2_P1-Bsl.npy')),0)
# G2_Pre_y2x = np.mean(np.squeeze(np.load('G2-4/Pre_y2x_G2_P1-Bsl.npy')),0)

# G3_Pre_x2y = np.mean(np.squeeze(np.load('G3-4/Pre_x2y_G3_P1-Bsl.npy')),0)
# G3_Pre_y2x = np.mean(np.squeeze(np.load('G3-4/Pre_y2x_G3_P1-Bsl.npy')),0)

# G4_Pre_x2y = np.mean(np.squeeze(np.load('G4-4/Pre_x2y_G4_P1-Bsl.npy')),0)
# G4_Pre_y2x = np.mean(np.squeeze(np.load('G4-4/Pre_y2x_G4_P1-Bsl.npy')),0)

# Stats_Sigs(G1_Pre_x2y, G1_Pre_y2x, 32,'gr-t_Pre_G1_P1-Bsl', 'gr')
# Stats_Sigs(G2_Pre_x2y, G2_Pre_y2x, 32,'gr-t_Pre_G2_P1-Bsl', 'gr')
# Stats_Sigs(G3_Pre_x2y, G3_Pre_y2x, 32,'gr-t_Pre_G3_P1-Bsl', 'gr')
# Stats_Sigs(G4_Pre_x2y, G4_Pre_y2x, 32,'gr-t_Pre_G4_P1-Bsl', 'gr')

# # # Pos
# G1_Pos_x2y = np.mean(np.squeeze(np.load('G1-4/Pos_x2y_G1_P1-Bsl.npy')),0)
# G1_Pos_y2x = np.mean(np.squeeze(np.load('G1-4/Pos_y2x_G1_P1-Bsl.npy')),0)

# G2_Pos_x2y = np.mean(np.squeeze(np.load('G2-4/Pos_x2y_G2_P1-Bsl.npy')),0)
# G2_Pos_y2x = np.mean(np.squeeze(np.load('G2-4/Pos_y2x_G2_P1-Bsl.npy')),0)

# G3_Pos_x2y = np.mean(np.squeeze(np.load('G3-4/Pos_x2y_G3_P1-Bsl.npy')),0)
# G3_Pos_y2x = np.mean(np.squeeze(np.load('G3-4/Pos_y2x_G3_P1-Bsl.npy')),0)

# G4_Pos_x2y = np.mean(np.squeeze(np.load('G4-4/Pos_x2y_G4_P1-Bsl.npy')),0)
# G4_Pos_y2x = np.mean(np.squeeze(np.load('G4-4/Pos_y2x_G4_P1-Bsl.npy')),0)

# Stats_Sigs(G1_Pos_x2y, G1_Pos_y2x, 20,'gr-t_Pos_G1_P1-Bsl', 'gr')
# Stats_Sigs(G2_Pos_x2y, G2_Pos_y2x, 20,'gr-t_Pos_G2_P1-Bsl', 'gr')
# Stats_Sigs(G3_Pos_x2y, G3_Pos_y2x, 20,'gr-t_Pos_G3_P1-Bsl', 'gr')
# Stats_Sigs(G4_Pos_x2y, G4_Pos_y2x, 20,'gr-t_Pos_G4_P1-Bsl', 'gr')


# # Frequency 
# # X to Y : V5 to V1
# G1_Pre_GrX = np.squeeze(np.load('G1-4/Pre_grx_G1_P1-Bsl.npy')) # frex x time-points
# G1_Pos_GrX = np.squeeze(np.load('G1-4/Pos_grx_G1_P1-Bsl.npy'))

# G2_Pre_GrX = np.squeeze(np.load('G2-4/Pre_grx_G2_P1-Bsl.npy')) # frex x time-points
# G2_Pos_GrX = np.squeeze(np.load('G2-4/Pos_grx_G2_P1-Bsl.npy'))

# G3_Pre_GrX = np.squeeze(np.load('G3-4/Pre_grx_G3_P1-Bsl.npy')) # frex x time-points
# G3_Pos_GrX = np.squeeze(np.load('G3-4/Pos_grx_G3_P1-Bsl.npy'))

# G4_Pre_GrX = np.squeeze(np.load('G4-4/Pre_grx_G4_P1-Bsl.npy')) # frex x time-points
# G4_Pos_GrX = np.squeeze(np.load('G4-4/Pos_grx_G4_P1-Bsl.npy'))

# Stats_TF(G1_Pre_GrX, G1_Pos_GrX, 'GrX_G1_P1-Bsl', 'GR')
# Stats_TF(G2_Pre_GrX, G2_Pos_GrX, 'GrX_G2_P1-Bsl', 'GR')
# Stats_TF(G3_Pre_GrX, G3_Pos_GrX, 'GrX_G3_P1-Bsl', 'GR')
# Stats_TF(G4_Pre_GrX, G4_Pos_GrX, 'GrX_G4_P1-Bsl', 'GR')

# # Y to X : V1 to V5
# G1_Pre_GrY = np.squeeze(np.load('G1-4/Pre_gry_G1_P1-Bsl.npy')) # frex x time-points
# G1_Pos_GrY = np.squeeze(np.load('G1-4/Pos_gry_G1_P1-Bsl.npy'))

# G2_Pre_GrY = np.squeeze(np.load('G2-4/Pre_gry_G2_P1-Bsl.npy')) # frex x time-points
# G2_Pos_GrY = np.squeeze(np.load('G2-4/Pos_gry_G2_P1-Bsl.npy'))

# G3_Pre_GrY = np.squeeze(np.load('G3-4/Pre_gry_G3_P1-Bsl.npy')) # frex x time-points
# G3_Pos_GrY = np.squeeze(np.load('G3-4/Pos_gry_G3_P1-Bsl.npy'))

# G4_Pre_GrY = np.squeeze(np.load('G4-4/Pre_gry_G4_P1-Bsl.npy')) # frex x time-points
# G4_Pos_GrY = np.squeeze(np.load('G4-4/Pos_gry_G4_P1-Bsl.npy'))

# Stats_TF(G1_Pre_GrY, G1_Pos_GrY, 'GrY_G1_P1-Bsl', 'GR')
# Stats_TF(G2_Pre_GrY, G2_Pos_GrY, 'GrY_G2_P1-Bsl', 'GR')
# Stats_TF(G3_Pre_GrY, G3_Pos_GrY, 'GrY_G3_P1-Bsl', 'GR')
# Stats_TF(G4_Pre_GrY, G4_Pos_GrY, 'GrY_G4_P1-Bsl', 'GR')


### Sources PLV

# G1_pre = np.empty([2,2,15,40,750])
# G1_pos = np.empty([2,2,15,40,750])
# G2_pre = np.empty([2,2,15,40,750])
# G2_pos = np.empty([2,2,15,40,750])
# G3_pre = np.empty([2,2,15,40,750])
# G3_pos = np.empty([2,2,15,40,750])
# G4_pre = np.empty([2,2,15,40,750])
# G4_pos = np.empty([2,2,15,40,750])
# G5_pre = np.empty([2,2,15,40,750])
# G5_pos = np.empty([2,2,15,40,750])

# for a in range(15):
    ##P10
#     G1_pre[:,:,a,:,:] = np.load('../../../Documents/09.19//Connectivity/Coherence/Data/P10_Bsl/Coh_%s_%s.npy' %('G1_Bsl', a+1))
#     G1_pos[:,:,a,:,:] = np.load('../../../Documents/09.19//Connectivity/Coherence/Data/P10_Bsl/Coh_%s_%s.npy' %('G1_P10', a+1))
#     G2_pre[:,:,a,:,:] = np.load('../../../Documents/09.19//Connectivity/Coherence/Data/P10_Bsl/Coh_%s_%s.npy' %('G2_Bsl', a+1))
#     G2_pos[:,:,a,:,:] = np.load('../../../Documents/09.19//Connectivity/Coherence/Data/P10_Bsl/Coh_%s_%s.npy' %('G2_P10', a+1))
#     G3_pre[:,:,a,:,:] = np.load('../../../Documents/09.19//Connectivity/Coherence/Data/P10_Bsl/Coh_%s_%s.npy' %('G3_Bsl', a+1))
#     G3_pos[:,:,a,:,:] = np.load('../../../Documents/09.19//Connectivity/Coherence/Data/P10_Bsl/Coh_%s_%s.npy' %('G3_P10', a+1))
#     G4_pre[:,:,a,:,:] = np.load('../../../Documents/09.19//Connectivity/Coherence/Data/P10_Bsl/Coh_%s_%s.npy' %('G4_Bsl', a+1))
#     G4_pos[:,:,a,:,:] = np.load('../../../Documents/09.19//Connectivity/Coherence/Data/P10_Bsl/Coh_%s_%s.npy' %('G4_P10', a+1))
#     G5_pre[:,:,a,:,:] = np.load('../../../Documents/09.19//Connectivity/Coherence/Data/P10_Bsl/Coh_%s_%s.npy' %('G5_Bsl', a+1))
#     G5_pos[:,:,a,:,:] = np.load('../../../Documents/09.19//Connectivity/Coherence/Data/P10_Bsl/Coh_%s_%s.npy' %('G5_P10', a+1))

   ##P30
#     G1_pre[:,:,a,:,:] = np.load('../../../Documents/09.19/Connectivity/Coherence/Data/P30_Bsl/Coh_%s_%s.npy' %('G1_Bsl', a+1))
#     G1_pos[:,:,a,:,:] = np.load('../../../Documents/09.19/Connectivity/Coherence/Data/P30_Bsl/Coh_%s_%s.npy' %('G1_P30', a+1))
#     G2_pre[:,:,a,:,:] = np.load('../../../Documents/09.19/Connectivity/Coherence/Data/P30_Bsl/Coh_%s_%s.npy' %('G2_Bsl', a+1))
#     G2_pos[:,:,a,:,:] = np.load('../../../Documents/09.19/Connectivity/Coherence/Data/P30_Bsl/Coh_%s_%s.npy' %('G2_P30', a+1))
#     G3_pre[:,:,a,:,:] = np.load('../../../Documents/09.19/Connectivity/Coherence/Data/P30_Bsl/Coh_%s_%s.npy' %('G3_Bsl', a+1))
#     G3_pos[:,:,a,:,:] = np.load('../../../Documents/09.19/Connectivity/Coherence/Data/P30_Bsl/Coh_%s_%s.npy' %('G3_P30', a+1))
#     G4_pre[:,:,a,:,:] = np.load('../../../Documents/09.19/Connectivity/Coherence/Data/P30_Bsl/Coh_%s_%s.npy' %('G4_Bsl', a+1))
#     G4_pos[:,:,a,:,:] = np.load('../../../Documents/09.19/Connectivity/Coherence/Data/P30_Bsl/Coh_%s_%s.npy' %('G4_P30', a+1))
#     G5_pre[:,:,a,:,:] = np.load('../../../Documents/09.19/Connectivity/Coherence/Data/P30_Bsl/Coh_%s_%s.npy' %('G5_Bsl', a+1))
#     G5_pos[:,:,a,:,:] = np.load('../../../Documents/09.19/Connectivity/Coherence/Data/P30_Bsl/Coh_%s_%s.npy' %('G5_P30', a+1))

# for b in range(2):
#     for c in range(b): #Change name for P10 vs P30
#         Stats_TF(G1_pre[b,c,:,:,:], G1_pos[b,c,:,:,:], '%s%s_Coh_G1_P3-Bsl' %(b,c), 'TF')
#         Stats_TF(G2_pre[b,c,:,:,:], G2_pos[b,c,:,:,:], '%s%s_Coh_G2_P3-Bsl' %(b,c), 'TF')
#         Stats_TF(G3_pre[b,c,:,:,:], G3_pos[b,c,:,:,:], '%s%s_Coh_G3_P3-Bsl' %(b,c), 'TF')
#         Stats_TF(G4_pre[b,c,:,:,:], G4_pos[b,c,:,:,:], '%s%s_Coh_G4_P3-Bsl' %(b,c), 'TF')
#         Stats_TF(G5_pre[b,c,:,:,:], G5_pos[b,c,:,:,:], '%s%s_Coh_G5_P3-Bsl' %(b,c), 'TF')

#         Stats_TF(G3_pre[b,c,:,:,:], G4_pre[b,c,:,:,:], '%s%s_PLV_G3G4_Bsl' %(b,c), 'TF')
#         Stats_TF(G3_pos[b,c,:,:,:], G4_pos[b,c,:,:,:], '%s%s_PLV_G4G3_P10' %(b,c), 'TF')



### PSD

# Analysis over Frequency bins

# Single Electrodes

# Pre_t = np.load('../../../Documents/09.19/Power/PSD_SingleElectrode/Pre_PSD_par_G2.npy')[:,1,:] # Subs x Chans x Freqs
# Pos_t = np.load('../../../Documents/09.19/Power/PSD_SingleElectrode/Pos_PSD_par_G2.npy')[:,1,:]
# Pre_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_SingleElectrode/Pre_PSD_par_G2.npy')[:,1,:],0)
# Pos_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_SingleElectrode/Pos_PSD_par_G2.npy')[:,1,:],0)

# Stats_Univar(Pre_t, Pos_t, Pre_t_avg, Pos_t_avg, 'P6_G2_P1-Bsl')

# All Electrodes Analysis

# P10
# G1_Pre_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G1_P1-Bsl.npy'),1)
# G1_Pos_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G1_P1-Bsl.npy'),1)
# G1_Pre_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G1_P1-Bsl.npy'),(0,1))
# G1_Pos_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G1_P1-Bsl.npy'),(0,1))

# G2_Pre_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G2_P1-Bsl.npy'),1)
# G2_Pos_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G2_P1-Bsl.npy'),1)
# G2_Pre_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G2_P1-Bsl.npy'),(0,1))
# G2_Pos_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G2_P1-Bsl.npy'),(0,1))

# G3_Pre_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G3_P1-Bsl.npy'),1)
# G3_Pos_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G3_P1-Bsl.npy'),1)
# G3_Pre_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G3_P1-Bsl.npy'),(0,1))
# G3_Pos_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G3_P1-Bsl.npy'),(0,1))

# G4_Pre_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G4_P1-Bsl.npy'),1)
# G4_Pos_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G4_P1-Bsl.npy'),1)
# G4_Pre_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G4_P1-Bsl.npy'),(0,1))
# G4_Pos_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G4_P1-Bsl.npy'),(0,1))

# G5_Pre_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G5_P1-Bsl.npy'),1)
# G5_Pos_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G5_P1-Bsl.npy'),1)
# G5_Pre_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G5_P1-Bsl.npy'),(0,1))
# G5_Pos_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G5_P1-Bsl.npy'),(0,1))

# Stats_Univar(G1_Pre_t, G1_Pos_t, G1_Pre_t_avg, G1_Pos_t_avg, 'G1_P1-Bsl_All')
# Stats_Univar(G2_Pre_t, G2_Pos_t, G2_Pre_t_avg, G2_Pos_t_avg, 'G2_P1-Bsl_All')
# Stats_Univar(G3_Pre_t, G3_Pos_t, G3_Pre_t_avg, G3_Pos_t_avg, 'G3_P1-Bsl_All')
# Stats_Univar(G4_Pre_t, G4_Pos_t, G4_Pre_t_avg, G4_Pos_t_avg, 'G4_P1-Bsl_All')
# Stats_Univar(G5_Pre_t, G5_Pos_t, G5_Pre_t_avg, G5_Pos_t_avg, 'G5_P1-Bsl_All')

# P30
# G1_Pre_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G1_P3-Bsl.npy'),1)
# G1_Pos_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G1_P3-Bsl.npy'),1)
# G1_Pre_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G1_P3-Bsl.npy'),(0,1))
# G1_Pos_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G1_P3-Bsl.npy'),(0,1))

# G2_Pre_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G2_P3-Bsl.npy'),1)
# G2_Pos_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G2_P3-Bsl.npy'),1)
# G2_Pre_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G2_P3-Bsl.npy'),(0,1))
# G2_Pos_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G2_P3-Bsl.npy'),(0,1))

# G3_Pre_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G3_P3-Bsl.npy'),1)
# G3_Pos_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G3_P3-Bsl.npy'),1)
# G3_Pre_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G3_P3-Bsl.npy'),(0,1))
# G3_Pos_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G3_P3-Bsl.npy'),(0,1))

# G4_Pre_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G4_P3-Bsl.npy'),1)
# G4_Pos_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G4_P3-Bsl.npy'),1)
# G4_Pre_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G4_P3-Bsl.npy'),(0,1))
# G4_Pos_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G4_P3-Bsl.npy'),(0,1))

# G5_Pre_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G5_P3-Bsl.npy'),1)
# G5_Pos_t = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G5_P3-Bsl.npy'),1)
# G5_Pre_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pre_PSD_All_G5_P3-Bsl.npy'),(0,1))
# G5_Pos_t_avg = np.mean(np.load('../../../Documents/09.19/Power/PSD_FullBrain/Pos_PSD_All_G5_P3-Bsl.npy'),(0,1))

# Stats_Univar(G1_Pre_t, G1_Pos_t, G1_Pre_t_avg, G1_Pos_t_avg, 'G1_P3-Bsl_All')
# Stats_Univar(G2_Pre_t, G2_Pos_t, G2_Pre_t_avg, G2_Pos_t_avg, 'G2_P3-Bsl_All')
# Stats_Univar(G3_Pre_t, G3_Pos_t, G3_Pre_t_avg, G3_Pos_t_avg, 'G3_P3-Bsl_All')
# Stats_Univar(G4_Pre_t, G4_Pos_t, G4_Pre_t_avg, G4_Pos_t_avg, 'G4_P3-Bsl_All')
# Stats_Univar(G5_Pre_t, G5_Pos_t, G5_Pre_t_avg, G5_Pos_t_avg, 'G5_P3-Bsl_All')

       
### Sources Cross_Frequency

# G1_pre = np.empty([15,5,6,2])
# G1_pos = np.empty([15,5,6,2])
# G2_pre = np.empty([15,5,6,2])
# G2_pos = np.empty([15,5,6,2])
# G3_pre = np.empty([15,5,6,2])
# G3_pos = np.empty([15,5,6,2])
# G4_pre = np.empty([15,5,6,2])
# G4_pos = np.empty([15,5,6,2])
# G5_pre = np.empty([15,5,6,2])
# G5_pos = np.empty([15,5,6,2])

# for a in range(15):
    ##P10 or P30
#     G1_pre[a,:,:,:] = np.load('../../../Documents/10.19/CrossFr_Sources/Data/zPAC/Bsl/zPAC_V1V5_%s_%s.npy' %('G1_Bsl', a+1))
#     G1_pos[a,:,:,:] = np.load('../../../Documents/10.19/CrossFr_Sources/Data/zPAC/P30/zPAC_V1V5_%s_%s.npy' %('G1_P30', a+1))
#     G2_pre[a,:,:,:] = np.load('../../../Documents/10.19/CrossFr_Sources/Data/zPAC/Bsl/zPAC_V1V5_%s_%s.npy' %('G2_Bsl', a+1))
#     G2_pos[a,:,:,:] = np.load('../../../Documents/10.19/CrossFr_Sources/Data/zPAC/P30/zPAC_V1V5_%s_%s.npy' %('G2_P30', a+1))
#     G3_pre[a,:,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/CrossFr_Src/Data/zPAC/Bsl/zPAC_V1V5_%s_%s.npy' %('G3_Bsl', a+1))
#     G3_pos[a,:,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/CrossFr_Src/Data/zPAC/P30/zPAC_V1V5_%s_%s.npy' %('G3_P30', a+1))
#     G4_pre[a,:,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/CrossFr_Src/Data/zPAC/Bsl/zPAC_V1V5_%s_%s.npy' %('G4_Bsl', a+1))
#     G4_pos[a,:,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/CrossFr_Src/Data/zPAC/P30/zPAC_V1V5_%s_%s.npy' %('G4_P30', a+1))
#     G5_pre[a,:,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/CrossFr_Src/Data/zPAC/Bsl/zPAC_V1V5_%s_%s.npy' %('G5_Bsl', a+1))
#     G5_pos[a,:,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/CrossFr_Src/Data/zPAC/P30/zPAC_V1V5_%s_%s.npy' %('G5_P30', a+1))
    
# print(G3_pre.shape) # Subjects x HiFr x LoFr x 2 Direction of Interaction
    
# for b in range(2): # b=0 V1pV5a, b=1 V1aV5p
#     Stats_CF(G1_pre[:,:,:,b], G1_pos[:,:,:,b], 'CF_%s_G1_P3~Bsl' %b)
#     Stats_CF(G2_pre[:,:,:,b], G2_pos[:,:,:,b], 'CF_%s_G2_P3~Bsl' %b)
#     Stats_CF(G3_pre[:,:,:,b], G3_pos[:,:,:,b], 'CF_%s_G3_P3~Bsl' %b)
#     Stats_CF(G4_pre[:,:,:,b], G4_pos[:,:,:,b], 'CF_%s_G4_P3~Bsl' %b)
#     Stats_CF(G5_pre[:,:,:,b], G5_pos[:,:,:,b], 'CF_%s_G5_P3~Bsl' %b)
##  #   Stats_TF(G1_pre[:,:,:,b], G1_pos[:,:,:,b], '%s_CF_G1_P1-Bsl' %b, 'CF')
    
# ## Between groups

# G1_Diff = G1_pos - G1_pre
# G2_Diff = G2_pos - G2_pre
# G5_Diff = G5_pos - G5_pre
# print(G1_Diff.shape)

# for b in range(2): # b=0 V1pV5a, b=1 V1aV5p
#     Stats_CF(G1_Diff[:,:,:,b], G2_Diff[:,:,:,b], '%s_G2-G1_P3-Bsl' %b)
#     Stats_CF(G1_Diff[:,:,:,b], G5_Diff[:,:,:,b], '%s_G5-G1_P3-Bsl' %b)
#     Stats_CF(G5_Diff[:,:,:,b], G2_Diff[:,:,:,b], '%s_G2-G5_P3-Bsl' %b)

# ## Different groups + same condition

# for b in range(2): # b=0 V1pV5a, b=1 V1aV5p
#     Stats_CF(G1_pre[:,:,:,b], G2_pre[:,:,:,b], '%s_G2-G1_Bsl'%b)
#     Stats_CF(G1_pos[:,:,:,b], G2_pos[:,:,:,b], '%s_G2-G1_P3' %b)
#     Stats_CF(G5_pre[:,:,:,b], G2_pre[:,:,:,b], '%s_G2-G5_Bsl'%b)
#     Stats_CF(G5_pos[:,:,:,b], G2_pos[:,:,:,b], '%s_G2-G5_P3' %b)
#     Stats_CF(G5_pre[:,:,:,b], G1_pre[:,:,:,b], '%s_G1-G5_Bsl'%b)
#     Stats_CF(G5_pos[:,:,:,b], G1_pos[:,:,:,b], '%s_G1-G5_P3' %b)
#     Stats_CF(G3_pre[:,:,:,b], G4_pre[:,:,:,b], '%s_G4-G3_Bsl'%b)
#     Stats_CF(G3_pos[:,:,:,b], G4_pos[:,:,:,b], '%s_G4-G3_P3' %b)
#     Stats_CF(G3_pre[:,:,:,b], G5_pre[:,:,:,b], '%s_G5-G3_Bsl'%b)
#     Stats_CF(G3_pos[:,:,:,b], G5_pos[:,:,:,b], '%s_G5-G3_P3' %b)
#     Stats_CF(G4_pre[:,:,:,b], G5_pre[:,:,:,b], '%s_G5-G4_Bsl'%b)
#     Stats_CF(G4_pos[:,:,:,b], G5_pos[:,:,:,b], '%s_G5-G4_P3' %b)




### Sources PSI /WPLI /Wpli

# For PSI change 40 Freq. bins for 4
G1_pre = np.empty([2,2,4,750,15])
G1_pos = np.empty([2,2,4,750,15])
G2_pre = np.empty([2,2,4,750,15])
G2_pos = np.empty([2,2,4,750,15])
G3_pre = np.empty([2,2,4,750,15])
G3_pos = np.empty([2,2,4,750,15])
G4_pre = np.empty([2,2,4,750,15])
G4_pos = np.empty([2,2,4,750,15])
G5_pre = np.empty([2,2,4,750,15])
G5_pos = np.empty([2,2,4,750,15])

for a in range(15):
    #P10
    G1_pre[:,:,:,:,a] = np.load('../../../Documents/1st_2nd_Results/Sources/Psi_Src/Psi_%s_%s.npy' %('G1_Bsl', a+1))
    G1_pos[:,:,:,:,a] = np.load('../../../Documents/1st_2nd_Results/Sources/Psi_Src/Psi_%s_%s.npy' %('G1_P30', a+1))
    G2_pre[:,:,:,:,a] = np.load('../../../Documents/1st_2nd_Results/Sources/Psi_Src/Psi_%s_%s.npy' %('G2_Bsl', a+1))
    G2_pos[:,:,:,:,a] = np.load('../../../Documents/1st_2nd_Results/Sources/Psi_Src/Psi_%s_%s.npy' %('G2_P30', a+1))
#     G3_pre[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G3_Bsl', a+1))
#     G3_pos[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G3_P10', a+1))
#     G4_pre[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G4_Bsl', a+1))
#     G4_pos[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G4_P10', a+1))
#     G5_pre[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G5_Bsl', a+1))
#     G5_pos[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G5_P10', a+1))

   ##P30
#     G1_pre[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G1_Bsl', a+1))
#     G1_pos[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G1_P30', a+1))
#     G2_pre[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G2_Bsl', a+1))
#     G2_pos[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G2_P30', a+1))
#     G3_pre[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G3_Bsl', a+1))
#     G3_pos[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G3_P30', a+1))
#     G4_pre[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G4_Bsl', a+1))
#     G4_pos[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G4_P30', a+1))
#     G5_pre[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G5_Bsl', a+1))
#     G5_pos[:,:,a,:,:] = np.load('../../../Documents/1st_2nd_Results/Sources/Wpli_Src/Wpli_%s_%s.npy' %('G5_P30', a+1))

for b in range(2):
    for c in range(b): #Change name for P10 vs P30
#         Stats_TF(G1_pre[b,c,:,:,:], G1_pos[b,c,:,:,:], '%s%s_Wpli_G1_P3-Bsl' %(b,c), 'TF')
#         Stats_TF(G2_pre[b,c,:,:,:], G2_pos[b,c,:,:,:], '%s%s_Wpli_G2_P3-Bsl' %(b,c), 'TF')
#         Stats_TF(G3_pre[b,c,:,:,:], G3_pos[b,c,:,:,:], '%s%s_Wpli_G3_P1_Bsl' %(b,c), 'TF')
#         Stats_TF(G4_pre[b,c,:,:,:], G4_pos[b,c,:,:,:], '%s%s_Wpli_G4_P1_Bsl' %(b,c), 'TF')
#         Stats_TF(G5_pre[b,c,:,:,:], G5_pos[b,c,:,:,:], '%s%s_Wpli_G5_P3_Bsl' %(b,c), 'TF')

        Stats_TF(G1_pos[b,c,:,:,:], G2_pos[b,c,:,:,:], '%s%s_Psi_G2G1_P3' %(b,c), 'TF')
#         Stats_TF(G3_pos[b,c,:,:,:], G5_pos[b,c,:,:,:], '%s%s_Wpli_G3G5_P3' %(b,c), 'TF')
#         Stats_TF(G4_pos[b,c,:,:,:], G5_pos[b,c,:,:,:], '%s%s_Wpli_G4G5_P3' %(b,c), 'TF')
