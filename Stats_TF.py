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

def Stats_TF(G1, G2, name, mode):
# G1: Pre-Stim. File with all subjects
# G2: Pos-Stim. File with all subjects
# name: name to save the output image
# mode: Either TF or GR analysis of the electrodes

    G1_avg = np.mean(G1,-1)
    G2_avg = np.mean(G2,-1)
    diffmap = np.subtract(G1_avg,G2_avg)
    print(G1_avg.shape, diffmap.shape)
#     diffmap = np.divide(G2_avg,G1_avg)

    
    if mode == 'TF':
        # Time Parameters TF
        min_time = -1.5 #-1.5
        max_time = 2.5  #1.5
        num_time = 1000 #750
        timex = np.linspace(min_time, max_time, num_time)
    elif mode == 'GR':
        # Time windows GR
        timex = np.linspace(-100, 700, num=32, endpoint=True)#ms
    elif mode == 'CF':
        # Time windows CF
        timex = np.linspace(3, 15, 6)#ms

    # Frequency Parameters : To be changed when CF
    min_freq = 1 #2
    max_freq = 99
    num_freq = 44  #40
#     frex = np.linspace(min_freq, max_freq, num_freq)
    frex = np.array([2,7,30,45]) # PSI
    
    ### Permutation test

    # P-val
    pval = 0.05

    # convert p-value to Z value
    zval = abs(st.norm.ppf(pval))

    # How many permutations?
    n_perms = 1000 #1000 for TimeFreq

    # Null hypothesis maps
    permmaps = np.zeros((n_perms, G1.shape[0], G1.shape[1]))
    print(permmaps.shape)

    # Power maps are concatenated
    tf3d = np.concatenate((G1,G2), axis=2)

    # Mapping the null hypothesis
    for p in range(n_perms):

        # randomize pixels, which also randomly assigns trials to channels
        randorder = np.random.permutation(tf3d.shape[2]);
        temp_tf3d = tf3d[:,:,randorder]

        # Difference map under the null hypothesis?
        permmaps[p,:,:] = np.squeeze(np.mean(temp_tf3d[:,:,0:G1.shape[2]],2) - np.mean(temp_tf3d[:,:,G1.shape[2]:tf3d.shape[2]+1],2) );

    # Non-corrected thresholded maps

    # compute mean and standard deviation maps
    mean_h0 = np.squeeze(np.mean(permmaps));
    std_h0  = np.squeeze(np.std(permmaps));

    # now threshold real data...
    # first Z-score
    zmap = (diffmap-mean_h0) / std_h0
    #print(zmap.shape)

    # threshold image at p-value, by setting subthreshold values to 0
    for x in range(zmap.shape[0]):
        for y in range(zmap.shape[1]):
            if abs(zmap[x,y]) < zval:
                zmap[x,y] = 0

    vmin= -0.01 #-0.12 PlV
    vmax= 0.01 #0.12 PLV
#     levels = MaxNLocator(nbins=40).tick_values(-1.25, 1.25) #PWR
    levels = MaxNLocator(nbins=40).tick_values(vmin, vmax) #PLV

#     fig1 = plt.figure(figsize=(25.0, 5.0)) # Horizontal fig (3,1,x)
    fig1 = plt.figure(figsize=(15.0, 10.0)) # Horizontal fig (3,1,x)
    ax1 = fig1.add_subplot(3,1,1)
    if mode == 'TF':
#         CS_1 = plt.contourf(timex[350:550], frex, diffmap[:,350:550], cmap='RdBu_r', levels=levels, norm=colors.Normalize(vmin=-1.25, vmax=1.25), extend='both') #PWR
        CS_1 = plt.contourf(timex[350:550], frex[:22], diffmap[:22,350:550], cmap='RdBu_r', levels=levels,
                            norm=colors.Normalize(vmin=vmin, vmax=vmax), extend='both') #PLV
        CS_1 =  plt.contour(timex[350:550], frex[:22], zmap[:,350:550], colors='red', levels=levels)#, norm=colors.Normalize(vmin=vmin, vmax=vmax))
    elif mode == 'GR':
        CS_1 = plt.contourf(timex, frex, diffmap)#, levels=levels, norm=colors.Normalize(vmin=-0.5, vmax=0.5))
        plt.contour(timex, frex, zmap, colors='red')#, levels=levels, norm=colors.Normalize(vmin=-0.5, vmax=0.5))
    elif mode == 'CF':
        CS_1 = plt.contourf(timex, frex, diffmap, cmap='RdBu_r', levels=levels, norm=colors.Normalize(vmin=vmin, vmax=vmax), extend='both') 
        plt.contour(timex, frex, zmap, colors='red', levels=levels)#, norm=colors.Normalize(vmin=vmin, vmax=vmax))
    cbar_1 = fig1.colorbar(CS_1, ax=ax1)
    ax1.set_title('Diff Pos-Pre')
    #ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')

    zmap_bin = np.where(abs(zmap)>0, 1, 0)
    ax2 = fig1.add_subplot(3,1,2)
    if mode == 'TF':
        CS_2 = plt.contourf(timex[365:525], frex[:22], zmap_bin[:22,365:525], cmap='RdBu_r', levels=levels)
    elif mode == 'GR':
        CS_2 = plt.contourf(timex, frex, zmap_bin, levels=levels)
    elif mode == 'CF':
        CS_2 = plt.contourf(timex, frex, zmap_bin, cmap='RdBu_r', levels=levels)
    cbar_2 = fig1.colorbar(CS_2, ax=ax2)
    ax2.set_title('Multiple Comparisons')
    #ax1.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')

    # Correction for multiple comparisons

    # Init. Maximum-pixel based correction --> "2" for min/max values for every pixel distribution
    max_val = np.zeros((n_perms,2))

    # Looping through permutations
    for pe in range(n_perms):

        # Get extreme values (smallest and largest)
        temp = np.sort(np.reshape(permmaps[pe,:,:],(1,permmaps.shape[1]*permmaps.shape[2])))
        max_val[pe,0] = temp.min()
        max_val[pe,1] = temp.max()

    # find the threshold for lower and upper values --> 2-tails test
    thresh_lo = np.percentile(max_val[:,0],    100*pval/2) 
    thresh_hi = np.percentile(max_val[:,1],100-100*pval/2) 

    # threshold real data
    zmap = diffmap*1
    for x in range(zmap.shape[0]):
        for y in range(zmap.shape[1]):
            print(zmap[x,y], thresh_lo, thresh_hi)
            if zmap[x,y]>thresh_lo and zmap[x,y]<thresh_hi:
                zmap[x,y] = 0

    zmap_bin_cor = np.where(abs(zmap)>0, 1, 0)
    ax3 = fig1.add_subplot(3,1,3)
    if mode == 'TF':
#         CS_3 = plt.contourf(timex[350:550], frex, zmap_bin_cor[:,350:550]) # PLV T.F.
        CS_3 = plt.contourf(timex[365:525], frex[:22], diffmap[:22,365:525], cmap='RdBu_r', levels=levels, extend='both') # Power T.F.
        plt.contour(timex[365:525], frex[:22], zmap_bin_cor[:22,365:525], colors='red', levels=levels)
    elif mode == 'GR':
        CS_3 = plt.contourf(timex, frex, zmap_bin_cor, levels=levels, norm=colors.Normalize(vmin=-1, vmax=1))
    elif mode == 'CF':
        CS_3 = plt.contourf(timex, frex, diffmap, cmap='RdBu_r', levels=levels, extend='both')
        plt.contour(timex, frex, zmap_bin_cor, colors='red', levels=levels)
    cbar_3 = fig1.colorbar(CS_3, ax=ax3)
    ax3.set_title('Corrected for Multiple Comparisons')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')
    
    fig1.savefig('Stats_TF_%s' %name)