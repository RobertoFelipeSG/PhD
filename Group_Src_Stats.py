import numpy as np
import pandas as pd
import scipy
import mne
import matplotlib
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
#matplotlib.use('TkAgg')

# name1 = ('G1_Bsl','G2_Bsl','G3_Bsl','G4_Bsl')
# name2 = ('G1_P10','G2_P10','G3_P10','G4_P10')
# Pre_con = np.empty([6,6,40,750,15,4])
# Pos_con = np.empty([6,6,40,750,15,4])

name1 = ('G3_Bsl','G4_Bsl')
name2 = ('G3_P10','G4_P10')
Pre_con = np.empty([2,2,40,750,15,2])
Pos_con = np.empty([2,2,40,750,15,2])

### Sham --> Implement at least 10 changes in the code
# name1 = ('G5_Bsl')
# name2 = ('G5_P10')
# Pre_con = np.empty([6,6,40,750,8,1])
# Pos_con = np.empty([6,6,40,750,8,1])

# Reading subject's data in Src.Space
for no, (n1, n2) in enumerate(zip(name1, name2)):
    for a in range(15):
        # 6 BrainAreas (3 RH + 3 LH) x 6 BrainAreas x Freqs.Bands x Timepoints x Subs x 10 Groups div. in name 1 or 2
        Pre_con[:,:,:,:,a,no] = np.load('../../../Documents/Sources_July19/Connectivity/Data/G%s/Coh_%s_%s.npy' %(no+3,n1, a+1))
        Pos_con[:,:,:,:,a,no] = np.load('../../../Documents/Sources_July19/Connectivity/Data/G%s/Coh_%s_%s.npy' %(no+3,n2, a+1))
#         Pos_con[:,:,:,:,a,no] = np.load('../../../Documents/Sources_July19/PLV/Data/G%s/PLV_%s_%s.npy' %(no+1,n2, a+1))
#     Pre_con[:,:,:,:,a,0] = np.load('../../../Documents/Sources_May19/PLV/Data/G5/PLV_%s_%s.npy' %(name1, a+1))
#     Pos_con[:,:,:,:,a,0] = np.load('../../../Documents/Sources_May19/PLV/Data/G5/PLV_%s_%s.npy' %(name2, a+1))
        
#print('he aqui', Pre_con[:,:,0:5,:,1])

##### ACTIVE GROUPS

gr = np.concatenate((Pre_con, Pos_con), axis=5)
print(gr.shape)

# Pairs = ([2,0],[3,0],[5,2]) # V1-MT, V1-V1, MT-MT
pairs = [1,0] # V1-MT

# maxx = np.empty([15,4,8,3]) # Subs. x Freqs.Bands x 8 Groups (name 1 + 2) x Pairwise PLV (e.g. V1-V2, V1-MT)
avgx = np.empty([15,4,8,1])

for g in range(gr.shape[5]): # 10 Groups (name 1 + 2)
    for sub in range(gr.shape[4]): # Subs.
#         for p, pr in enumerate(Pairs): # Pairwise PLV
#             maxx[sub,0,g,p] = np.amax(np.mean(gr[pr[0],pr[1],0:5,375:500,sub,g], axis=0), axis=0)
#             maxx[sub,1,g,p] = np.amax(np.mean(gr[pr[0],pr[1],5:12,375:500,sub,g], axis=0), axis=0)
#             maxx[sub,2,g,p] = np.amax(np.mean(gr[pr[0],pr[1],12:29,375:500,sub,g], axis=0), axis=0)
#             maxx[sub,3,g,p] = np.amax(np.mean(gr[pr[0],pr[1],29:40,375:500,sub,g], axis=0), axis=0)
#             avgx[sub,0,g,p] = np.mean(np.mean(gr[pr[0],pr[1],0:5,375:500,sub,g], axis=0), axis=0)
        avgx[sub,0,g,0] = np.mean(np.mean(gr[pairs[0],pairs[1],0:5,375:500,sub,g], axis=0), axis=0)
        avgx[sub,1,g,0] = np.mean(np.mean(gr[pairs[0],pairs[1],5:12,375:500,sub,g], axis=0), axis=0)
        avgx[sub,2,g,0] = np.mean(np.mean(gr[pairs[0],pairs[1],12:29,375:500,sub,g], axis=0), axis=0)
        avgx[sub,3,g,0] = np.mean(np.mean(gr[pairs[0],pairs[1],29:40,375:500,sub,g], axis=0), axis=0)
 
print('size avgx')
print(avgx.shape)  # Subs. x Freqs.Bands x Groups (Pre vs Pos) x Pairwise PLV (e.g. V1-V2, V1-MT)
        
#### To a PD DataFrame      

# label_Pre_max = np.array(['The_Pre_max_V1-V2', 'Alp_Pre_max_V1-V2', 'Bet_Pre_max_V1-V2', 'Gam_Pre_max_V1-V2',
#                           'The_Pre_max_V2-MT', 'Alp_Pre_max_V2-MT', 'Bet_Pre_max_V2-MT', 'Gam_Pre_max_V2-MT',
#                           'The_Pre_max_V1-MT', 'Alp_Pre_max_V1-MT', 'Bet_Pre_max_V1-MT', 'Gam_Pre_max_V1-MT'])
# label_Pos_max = np.array(['The_Pos_max_V1-V2', 'Alp_Pos_max_V1-V2', 'Bet_Pos_max_V1-V2', 'Gam_Pos_max_V1-V2',
#                           'The_Pos_max_V2-MT', 'Alp_Pos_max_V2-MT', 'Bet_Pos_max_V2-MT', 'Gam_Pos_max_V2-MT',
#                           'The_Pos_max_V1-MT', 'Alp_Pos_max_V1-MT', 'Bet_Pos_max_V1-MT', 'Gam_Pos_max_V1-MT'])
# Eeg_Pre_max = pd.DataFrame(data=(maxx[:,:,0:4,:].reshape((60,12))), columns=label_Pre_max)
# Eeg_Pos_max = pd.DataFrame(data=(maxx[:,:,4:8,:].reshape((60,12))), columns=label_Pos_max)

label_Pre_avg = np.array(['The_Pre_avg_V1-MT', 'Alp_Pre_avg_V1-MT', 'Bet_Pre_avg_V1-MT', 'Gam_Pre_avg_V1-MT'])
#                           'The_Pre_avg_V1-V1', 'Alp_Pre_avg_V1-V1', 'Bet_Pre_avg_V1-V1', 'Gam_Pre_avg_V1-V1',
#                           'The_Pre_avg_MT-MT', 'Alp_Pre_avg_MT-MT', 'Bet_Pre_avg_MT-MT', 'Gam_Pre_avg_MT-MT'])
label_Pos_avg = np.array(['The_Pos_avg_V1-MT', 'Alp_Pos_avg_V1-MT', 'Bet_Pos_avg_V1-MT', 'Gam_Pos_avg_V1-MT'])
#                           'The_Pos_avg_V1-V1', 'Alp_Pos_avg_V1-V1', 'Bet_Pos_avg_V1-V1', 'Gam_Pos_avg_V1-V1',
#                           'The_Pos_avg_MT-MT', 'Alp_Pos_avg_MT-MT', 'Bet_Pos_avg_MT-MT', 'Gam_Pos_avg_MT-MT'])

# 60x12 = Subs. x Freqs.Bands x Groups (Pre vs Pos) x Pairwise PLV (e.g. V1-V2, V1-MT)
# Eeg_Pre_avg = pd.DataFrame(data=(avgx[:,:,0:4,:].reshape((60,12))), columns=label_Pre_avg) 
Eeg_Pre_avg = pd.DataFrame(data=(avgx[:,:,0:2,:].reshape((30,4))), columns=label_Pre_avg) 
Eeg_Pos_avg = pd.DataFrame(data=(avgx[:,:,2:4,:].reshape((30,4))), columns=label_Pos_avg)

# combined = pd.concat([Eeg_Pre_max,Eeg_Pos_max, Eeg_Pre_avg, Eeg_Pos_avg], axis=1)
combined = pd.concat([Eeg_Pre_avg, Eeg_Pos_avg], axis=1)
# print(combined)
combined.to_csv('Avg_Coh_Gr.txt', sep='\t')


      

