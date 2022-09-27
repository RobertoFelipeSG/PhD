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

### Power

# G1_ver = np.empty([44,1000,15])
# G1_sh = np.empty([44,1000,15])
# G2_ver = np.empty([44,1000,15])
# G2_sh = np.empty([44,1000,15])
# G1_cau = np.empty([44,1000,30]) # Change if Verum x2
# G2_cau = np.empty([44,1000,30])

# for a in range(30): # Change if Verum x2
#     G1_ver[:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/TW/Pwr_%s_%s.npy' %('G1_Bursts', a+1))
#     G1_sh[:,:,a]  = np.load('../../../Documents/3rd_Results/Subjects/TW/Pwr_%s_%s.npy' %('G1_BSham', a+1))
#     G2_ver[:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/TW/Pwr_%s_%s.npy' %('G2_Bursts', a+1))
#     G2_sh[:,:,a]  = np.load('../../../Documents/3rd_Results/Subjects/TW/Pwr_%s_%s.npy' %('G2_BSham', a+1))
#     if a <= 14:
#         G1_cau[:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/TW/Pwr_%s_Bursts_%s.npy' %('G1', a+1))
#         G2_cau[:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/TW/Pwr_%s_BSham_%s.npy' %('G1', a+1))
#     elif a > 14:
#         G1_cau[:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/TW/Pwr_%s_Bursts_%s.npy' %('G2', a-14)) # Change if Verum x2
#         G2_cau[:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/TW/Pwr_%s_BSham_%s.npy' %('G2', a-14))
        
# Stats_TF(G1_ver[:,:,:], G1_sh[:,:,:], 'Pwr_G1_Bursts_G1_BSham', 'TF')
# Stats_TF(G2_ver[:,:,:], G2_sh[:,:,:], 'Pwr_G2_Bursts_G2_BSham', 'TF')
# Stats_TF(G1_ver[:,:,:], G2_ver[:,:,:], 'Pwr_G2-G1_Ver', 'TF')
# Stats_TF(G1_sh[:,:,:], G2_sh[:,:,:],  'Pwr_G2-G1_Sh', 'TF')

# Stats_TF(G1_cau[:,:,:], G2_cau[:,:,:], 'Pwr_Ver-Sh', 'TF')

# Stats_TF(G1_cau[:,:,:], G2_cau[:,:,:], 'ITPC_Ver-Sh', 'TF')   # Change if Verum x2

    
### ITPC

# G1_ver = np.empty([2,44,1000,15], dtype=complex)
# G1_sh = np.empty([2,44,1000,15], dtype=complex)
# G2_ver = np.empty([2,44,1000,15], dtype=complex)
# G2_sh = np.empty([2,44,1000,15], dtype=complex)
# G1_cau = np.empty([2,44,1000,30], dtype=complex) # Change if Verum x2
# G2_cau = np.empty([2,44,1000,30], dtype=complex)

# for a in range(15): # Change to 30 if Verum x2
#     G1_ver[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/TW/Wave_%s_%s.npy' %('G1_Bursts', a+1))
#     G1_sh[:,:,:,a]  = np.load('../../../Documents/3rd_Results/Subjects/TW/Wave_%s_%s.npy' %('G1_BSham', a+1))
#     G2_ver[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/TW/Wave_%s_%s.npy' %('G2_Bursts', a+1))
#     G2_sh[:,:,:,a]  = np.load('../../../Documents/3rd_Results/Subjects/TW/Wave_%s_%s.npy' %('G2_BSham', a+1))
#     if a <= 14:
#         G1_cau[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/TW/Wave_%s_Bursts_%s.npy' %('G1', a+1))
#         G2_cau[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/TW/Wave_%s_BSham_%s.npy' %('G1', a+1))
#     elif a > 14:
#         G1_cau[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/TW/Wave_%s_Bursts_%s.npy' %('G2', a-14)) # Change if Verum x2
#         G2_cau[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/TW/Wave_%s_BSham_%s.npy' %('G2', a-14))
        
# Avg over channels V1-V5
# G1_itpc_ver = np.mean(abs(G1_ver),0)     # Input: Ch x frex x time-points x Sub
# G1_itpc_sh = np.mean(abs(G1_sh),0)       # Output: frex x time-points x Sub
# G2_itpc_ver = np.mean(abs(G2_ver),0)     
# G2_itpc_sh = np.mean(abs(G2_sh),0) 

# Stats_TF(G1_itpc_ver[:,:,:], G1_itpc_sh[:,:,:], 'ITPC_G1_Bursts_G1_BSham', 'TF')
# Stats_TF(G2_itpc_ver[:,:,:], G2_itpc_sh[:,:,:], 'ITPC_G2_Bursts_G2_BSham', 'TF')
# Stats_TF(G1_itpc_ver[:,:,:], G2_itpc_ver[:,:,:], 'ITPC_G2-G1_Ver', 'TF')
# Stats_TF(G1_itpc_sh[:,:,:], G2_itpc_sh[:,:,:],  'ITPC_G2-G1_Sh', 'TF')

# Avg over channels V1-V5
# G1_itpc_cau = np.mean(abs(G1_cau),0)     
# G2_itpc_cau = np.mean(abs(G2_cau),0)

# Stats_TF(G1_itpc_cau[:,:,:], G2_itpc_cau[:,:,:], 'ITPC_G2-Sh', 'TF')

# Stats_TF(G1_itpc_cau[:,:,:], G2_itpc_cau[:,:,:], 'ITPC_Ver-Sh', 'TF')   # Change if Verum x2


### Sources Cross_Frequency

G1_ver = np.empty([5,6,2,15])
# G1_sh = np.empty([5,6,2,30])
G2_ver = np.empty([5,6,2,15])
# G2_sh = np.empty([5,6,2,30])

for a in range(15): # Change to 30 if Verum x2
    G1_ver[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_%s.npy' %('G1_Bursts', a+1))
#     G1_sh[:,:,:,a]  = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_%s.npy' %('G1_BSham', a+1))
    G2_ver[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_%s.npy' %('G2_Bursts', a+1))
#     G2_sh[:,:,:,a]  = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_%s.npy' %('G2_BSham', a+1))
#     if a <= 14:
#         G1_ver[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_Bursts_%s.npy' %('G1', a+1)) #Verum InP
#         G2_sh[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_BSham_%s.npy' %('G1', a+1))   #Sham InP
#     elif a > 14:
#         G1_ver[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_Bursts_%s.npy' %('G2', a-14)) #Verum AnP
#         G2_sh[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_BSham_%s.npy' %('G2', a-14))  #Sham AnP
        
for b in range(2): # b=0 V1pV5a, b=1 V1aV5p
#     Stats_CF(G1_ver[:,:,b,:], G2_sh[:,:,b,:], 'CF_%s_G2-Sh' %b)
    # Between Verum Groups
    Stats_CF(G1_ver[:,:,b,:], G2_ver[:,:,b,:], 'CF_%s_G2-G1_Ver' %b)

    


### Sources PSI /WPLI /Wpli

# G1_ver = np.empty([2,2,44,1000,15])
# G1_sh = np.empty([2,2,44,1000,15])
# G2_ver = np.empty([2,2,44,1000,15])
# G2_sh = np.empty([2,2,44,1000,15])
# G1_cau = np.empty([2,2,4,1000,35])
# G2_cau = np.empty([2,2,4,1000,30])

# for a in range(30):
#     G1_ver[:,:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/Wpli/WPLI_%s_%s.npy' %('G1_Bursts', a+1))
#     G1_sh[:,:,:,:,a]  = np.load('../../../Documents/3rd_Results/Subjects/Wpli/WPLI_%s_%s.npy' %('G1_BSham', a+1))
#     G2_ver[:,:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/Wpli/WPLI_%s_%s.npy' %('G2_Bursts', a+1))
#     G2_sh[:,:,:,:,a]  = np.load('../../../Documents/3rd_Results/Subjects/Wpli/WPLI_%s_%s.npy' %('G2_BSham', a+1))
#     if a <= 14:
#         G1_cau[:,:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/Psi/PSI_%s_Bursts_%s.npy' %('G1', a+1))
#         G2_cau[:,:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/Psi/PSI_%s_BSham_%s.npy' %('G1', a+1))
#     elif a > 14:
#         G1_cau[:,:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/Psi/PSI_%s_Bursts_%s.npy' %('G2', a-14))
#         G2_cau[:,:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/Psi/PSI_%s_BSham_%s.npy' %('G2', a-14))

# for b in range(2):
#     for c in range(b): 
#         Stats_TF(G1_ver[b,c,:,:,:], G1_sh[b,c,:,:,:], '%s%s_WPLI_G1_Bursts_G1_BSham' %(b,c), 'TF')
#         Stats_TF(G2_ver[b,c,:,:,:], G2_sh[b,c,:,:,:], '%s%s_WPLI_G2_Bursts_G2_BSham' %(b,c), 'TF')

# Separated groups vs. Sham / Between groups
#         Stats_TF(G1_cau[b,c,:,:,:], G2_cau[b,c,:,:,:], '%s%s_PSI_G2-Sh' %(b,c), 'TF')
#         Stats_TF(G1_ver[b,c,:,:,:], G2_ver[b,c,:,:,:], '%s%s_WPLI_G2-G1_Ver' %(b,c), 'TF')
#         Stats_TF(G1_sh[b,c,:,:,:],  G2_sh[b,c,:,:,:], '%s%s_WPLI_G2-G1_Sh' %(b,c), 'TF')

# Groups packed together
#         Stats_TF(G1_cau[b,c,:,:,:], G2_cau[b,c,:,:,:], '%s%s_PSI_Ver-Sh' %(b,c), 'TF')



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





       

