import numpy as np
import mne
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from mne.stats import fdr_correction, bonferroni_correction

def Stats_Sigs(G1, G2, numbins, name, mode):
# G1: Pre-Stim. File with all subjects
# G2: Pos-Stim. File with all subjects
# numbins: Number of windows where to perform a T-test
# name: name to save the output image
# mode: Either 'time', 'freq' or 'gr' analysis of the electrodes


    # Time Parameters TF
    min_time = -1500
    max_time = 1500
    num_time = 750
    timex = np.linspace(min_time, max_time, num_time)

    # Frequency Parameters
    min_freq = 2
    max_freq = 42
    num_freq = 40
    frex = np.linspace(min_freq, max_freq, num_freq)
    
    # Granger Time windows 
    timegr = np.linspace(-100, 700, num=32, endpoint=True)#ms

    bins = np.empty((numbins,2))

    # Binning for Multiple comparisons
    if mode == 'time':
        val_max = 3000
        aux_bins = np.ceil(np.linspace(0, val_max, numbins))
    elif mode == 'freq':
        val_max = 42
        aux_bins = np.ceil(np.linspace(2, val_max, numbins))
    elif mode == 'gr':
        val_max = 700
        aux_bins = np.ceil(np.linspace(-100, val_max, numbins))
    
    print(aux_bins)
    for a, b in enumerate(aux_bins):
        bins[a, 1] = np.ceil((b * G1.shape[0]) / val_max)
    bins = bins.astype(int)
    bins[1:,0] = [bins[c,1] for c,_ in enumerate(bins[1:,1])]
    print (bins)

    # Sliding T-test
    sig = np.empty([numbins,2])
    pvalues = np.zeros([numbins])
    for idx,_ in enumerate(bins):
        [stat, pval] = stats.ttest_rel(G1[bins[idx,0]:bins[idx,1]], G2[bins[idx,0]:bins[idx,1]])
        if pval < 0.01:
            pvalues[idx] = pval
            sig[idx,:] = np.array([bins[idx,0], bins[idx,1]])
        else:
            pvalues[idx] = pval
            sig[idx,:] = 0

    pv = np.isnan(pvalues)
    pvalues[pv] = 100
    #print()
    reject_H0, fdr_pvals = fdr_correction(pvalues, 0.01) #False Discovery Rate
    #print(reject_H0)
    fdr = np.where(reject_H0 == True)
    #print(sig)
    #print(sig[fdr])

    fig1 = plt.figure(figsize=(13.0, 7.5))
    if mode == 'time':
        #ax1 = fig2.add_subplot(2,1,1)
        stdev1 = np.std(G1)
        plt.plot(timex[350:550], G1[350:550], label='Pre')
        plt.tick_params(labelsize=20)
        plt.fill_between(timex[350:550], G1[350:550]+stdev1, G1[350:550]-stdev1, alpha=.1)
        stdev2 = np.std(G2)
        plt.plot(timex[350:550], G2[350:550], label='Pos') 
        plt.tick_params(labelsize=20)
        plt.fill_between(timex[350:550], G2[350:550]+stdev2, G2[350:550]-stdev2, alpha=.1)
        for s in sig[fdr]:
            plt.fill_between(s, 0, 1, color='lightgray')
            plt.axvline(s[0], color='r') # Show Stim Onset
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.ylabel('Pair Phase Synchrony - PLV', fontsize=20)
        plt.xlabel('Time(s)', fontsize=20)
        plt.ylim([0,0.6])
        fig1.savefig('Stats_PLV_%s' %name)

    if mode == 'freq':    
        stdev3 = np.std(G1)
        plt.plot(frex, G1, label='Pre')
        plt.tick_params(labelsize=20)
        plt.fill_between(frex, G1+stdev3, G1-stdev3, alpha=.1)
        plt.plot(frex, G2, label='Pos')
        stdev4 = np.std(G2)
        plt.tick_params(labelsize=20)
        plt.fill_between(frex, G2+stdev4, G2-stdev4, alpha=.1)
        for s in sig[fdr]:
            plt.fill_between(s, 0, 1, color='lightgray')
        plt.tick_params(labelsize=20)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.ylabel('Pair Phase Synchrony - PLV', fontsize=20)
        plt.xlabel('Frequency(Hz)', fontsize=20)
        plt.ylim([0,0.4])
        fig1.savefig('Stats_PLV_%s' %name)
        
    if mode == 'gr':
        stdev5 = np.std(G1)
        plt.plot(timegr, G1, label='v5 to v1')
        plt.tick_params(labelsize=20)
        plt.fill_between(timegr, G1+stdev5, G1-stdev5, alpha=.1)
        stdev6 = np.std(G2)
        plt.plot(timegr, G2, label='v1 to v5')
        plt.tick_params(labelsize=20)
        plt.fill_between(timegr, G2+stdev6, G2+stdev6, alpha=.1)
        for s in sig[fdr]:
            plt.fill_between(s, 0, 1, color='lightgray')
        plt.ylim([0.0,0.01])
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.xlabel('Time (s)')
        plt.ylabel('G-causality')
        fig1.savefig('Stats_%s' %name)