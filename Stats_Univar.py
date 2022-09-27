import numpy as np
import scipy, scipy.io
import mne
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import pandas as pd
from scipy import fft, ifft, arange
import scipy.stats as st
from scipy import stats
from mne.stats import fdr_correction, bonferroni_correction

def Stats_Univar(G1, G2, G1_avg, G2_avg, name):#, mode):
# G1: Pre-Stim. File with all subjects
# G2: Pos-Stim. File with all subjects
# G1_avg: Pre-Stim. File with mean of subejcts
# G2_avg: Pos-Stim. File with mean of subejcts
# name: name to save the output image
# mode: Either TF or GR analysis of the electrodes

#     # Time Parameters TF
#     min_time = -1500
#     max_time = 1500
#     num_time = 750
#     timex = np.linspace(min_time, max_time, num_time)
    
    # Time Parameters TF
    min_time = 0
    max_time = 125
    num_time = 125
    timex = np.linspace(min_time, max_time, num_time)

#     numbins = 750
    numbins = 125

    print(G1.shape)
    
    # Sliding T-test
    sig = np.empty([numbins,2])
    pvalues = np.zeros([numbins])

    for idx in range(numbins):
#         [stat, pval] = stats.ttest_rel(np.mean(G1[1,0,:,5:11,idx],1), np.mean(G2[1,0,:,5:11,idx],1),0)
        [stat, pval] = stats.wilcoxon(G1[:,idx], G2[:,idx])#,0)
        if pval < 0.02:
            pvalues[idx] = pval
            sig[idx,:] = np.array([idx, idx+1])
        else:
            pvalues[idx] = 0
            sig[idx,:] = 0
#     print(np.nonzero(sig))
    print(sig)

    reject_H0, fdr_pvals = fdr_correction(pvalues, 0.02) #False Discovery Rate
    #print(reject_H0)
    fdr = np.where(reject_H0 == True)
    # print(np.nonzero(sig[fdr]))

#     fig1 = plt.figure(figsize=(13.0, 7.5))
#     #ax1 = fig2.add_subplot(2,1,1)
#     stdev1 = np.std(G1_avg)
#     plt.plot(timex[350:550], G1_avg[350:550], label='Pre')
#     plt.tick_params(labelsize=20)
#     plt.fill_between(timex[350:550], G1_avg[350:550]+stdev1, G1_avg[350:550]-stdev1, alpha=.1)
#     stdev2 = np.std(G2_avg)
#     plt.plot(timex[350:550], G2_avg[350:550], label='Pos') 
#     plt.tick_params(labelsize=20)
#     plt.fill_between(timex[350:550], G2_avg[350:550]+stdev2, G2_avg[350:550]-stdev2, alpha=.1)
#     for s in sig[fdr]:
#         plt.fill_between(s, 0, 1, color='lightgray')
#     #     plt.axvline(s[0], color='r') # Show Stim Onset
#     plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
#     plt.ylabel('Pair Phase Synchrony - PLV', fontsize=20)
#     plt.xlabel('Time(s)', fontsize=20)
#     plt.ylim([0,0.3])
#     fig1.savefig('Stats_PLV_%s' %name)

    fig1 = plt.figure(figsize=(13.0, 7.5))
    #ax1 = fig2.add_subplot(2,1,1)
    stdev1 = np.std(G1_avg)
    plt.plot(timex, G1_avg.T, label='Pre')
    plt.tick_params(labelsize=20)
#     plt.fill_between(timex, G1_avg+stdev1, G1_avg-stdev1, alpha=.1)
    stdev2 = np.std(G2_avg)
    plt.plot(timex, G2_avg.T, label='Pos') 
    plt.tick_params(labelsize=20)
#     plt.fill_between(timex, G2_avg+stdev2, G2_avg-stdev2, alpha=.1)
    for s in sig[fdr]:
        plt.fill_between(s, 0, 1, color='lightgray')
    #     plt.axvline(s[0], color='r') # Show Stim Onset
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.ylabel('PSD', fontsize=20)
    plt.xlabel('Freq', fontsize=20)
    plt.ylim([0,6e-12])
    plt.xlim([0, 45])
    fig1.savefig('Stats_PSD_%s' %name)