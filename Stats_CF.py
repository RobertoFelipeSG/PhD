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
from skimage import measure
from skimage.measure import label, regionprops

def Stats_CF(G1, G2, name):
# G1: Pre-Stim. File with all subjects
# G2: Pos-Stim. File with all subjects
# name: name to save the output image
# mode: Either TF or GR analysis of the electrodes

    G1_avg = np.mean(G1,-1)
    G2_avg = np.mean(G2,-1)
    diffmap = np.subtract(G2_avg,G1_avg)
#     diffmap = np.divide(G2_avg,G1_avg)
    print(G1_avg.shape, diffmap.shape)

    # Lo Frequency Parameters
    timex = np.linspace(3, 15, 6)

    # Hi Frequency Parameters
    frex = np.linspace(22, 42, 5)
    
    # Permutation test
    # P-val
    pval = 0.05

    # convert p-value to Z value
    zval = abs(st.norm.ppf(pval))

    # How many permutations?
    n_perms = 1000

    # Null hypothesis maps
    permmaps = np.zeros((n_perms, G1.shape[0], G1.shape[1]))

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
    print(zmap.shape)

    # threshold image at p-value, by setting subthreshold values to 0
    for x in range(zmap.shape[0]):
        for y in range(zmap.shape[1]):
            if abs(zmap[x,y]) < zval:
                zmap[x,y] = 0

#     vmin= -0.75 
#     vmax= 0.75 #Difference
    vmin= -1 
    vmax= 1 #Ratio 
    levels = MaxNLocator(nbins=60).tick_values(vmin, vmax) #PLV

    fig1 = plt.figure(figsize=(15.0,4.0))
    ax1 = fig1.add_subplot(1,3,1)
    CS_1 = plt.contourf(timex, frex, diffmap, cmap='RdBu_r', levels=levels, norm=colors.Normalize(vmin=vmin, vmax=vmax), extend='both') 
    plt.contour(timex, frex, zmap, colors='red', levels=levels, norm=colors.Normalize(vmin=vmin, vmax=vmax))
    cbar_1 = fig1.colorbar(CS_1, ax=ax1)
    ax1.set_title('Diff Pos-Pre')
    #ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Frequency (Hz)')

    zmap_bin = np.where(abs(zmap)>0, 1, 0)
    ax2 = fig1.add_subplot(1,3,2)
    CS_2 = plt.contourf(timex, frex, zmap_bin, cmap='RdBu_r')#, levels=levels)
    cbar_2 = fig1.colorbar(CS_2, ax=ax2)
    ax2.set_title('Multiple Comparisons')
    #ax1.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')

    # Correction for multiple comparisons

    # Init. Cluster based correction 
    max_cluster_sizes = np.zeros((n_perms,1))
    num = np.zeros((n_perms,1))

    # Looping through permutations
    for pe in range(n_perms):
            
        threshing = np.squeeze(permmaps[pe,:,:].astype(int))
        label_thresh = label(threshing)
        islands, num[pe] = measure.label(label_thresh, return_num=True) # Finding and measuring the clusters
        prop = regionprops(islands)
        for pr in prop:
            if pr.area > 0:        
                max_cluster_sizes[pe] = np.max(pr.area);
         
    # based on p-value and null hypothesis distribution
    cluster_thresh = np.percentile(max_cluster_sizes,100-(100*pval), interpolation='midpoint')
    print("cluster_thresh", cluster_thresh)

    # threshold real data
    island, nu = measure.label(zmap, return_num=True)
    print('numb_label', nu)
    props = regionprops(island)
    # if real clusters are too small, remove them by setting them to zero!
    for n in range(nu):        
        for pro in props:
            if pro.area < cluster_thresh:
                zmap[pro.area]=0

    zmap_bin_cor = np.where(abs(zmap)>0, 1, 0)
    
    levels1 = MaxNLocator(nbins=2).tick_values(vmin, vmax) #Modify to enhance significance lines
    
    ax3 = fig1.add_subplot(1,3,3)
    CS_3 = plt.contourf(timex, frex, diffmap, cmap='RdBu_r', levels=levels, extend='both')
    plt.contour(timex, frex, zmap_bin_cor, colors='red', levels=levels1)
    cbar_3 = fig1.colorbar(CS_3, ax=ax3)
#     ax3.tick_params(labelsize=20)
#     ax3.set_ylabel('Phase', fontsize=20)
#     ax3.set_xlabel('Time(s)', fontsize=20)
    ax3.set_title('Corrected for Multiple Comparisons')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Frequency (Hz)')
    
    fig1.savefig('Stats_%s' %name)