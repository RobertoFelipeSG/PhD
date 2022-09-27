import numpy as np
import scipy
import mne
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

#matplotlib.use('TkAgg')

from GrangerCaus import *

def Group_Gr (G1, G2, type_tr1, type_tr2, chans_x, chans_y, name):
# G1: list of paths corresponding to 1st group
# G2: list of paths corresponding to 2nd group
# type_tr: Correct='stim_cor' or Incorrect='stim_inc'
# chans_x: Channels to be averaged as X in Granger 
# chans_y: Channels to be averaged as Y in Granger
# name: name to save the output image

    # Time windows to evaluate
    times2 = np.linspace(-100, 700, num=32, endpoint=True)#ms
    times2save = np.empty((1,times2.size))
    times2save[0,:] = times2
    print(times2save.shape)
    
    # Frequency Parameters
    min_freq = 2
    max_freq = 42
    num_freq = 40
    frex = np.linspace(min_freq, max_freq, num_freq)

    # Initialize Group Arrays for Granger
    G1_x2y = np.empty((len(G1),1,times2.size))
    G1_y2x = np.empty((len(G1),1,times2.size))
    G2_x2y = np.empty((len(G2),1,times2.size))
    G2_y2x = np.empty((len(G2),1,times2.size))
    G1_tf_gr = np.empty((len(G1),2,40,times2.size))
    G2_tf_gr = np.empty((len(G2),2,40,times2.size))
    
    # Looping over subjects
    for id1, sub1 in enumerate(G1):
        G1_y2x[id1,:,:], G1_x2y[id1,:,:], G1_tf_gr[id1,:,:,:] = GrangerCaus(sub1, type_tr1, chans_x, chans_y)
         
    for id2, sub2 in enumerate(G2):
        G2_y2x[id2,:,:], G2_x2y[id2,:,:], G2_tf_gr[id2,:,:,:] = GrangerCaus(sub2, type_tr2, chans_x, chans_y)
          
    # Saving Data for stats
    np.save('Pre_y2x_%s' %name, G1_y2x)
    np.save('Pos_y2x_%s' %name, G2_y2x)
    np.save('Pre_x2y_%s' %name, G1_x2y)
    np.save('Pos_x2y_%s' %name, G2_x2y)
    np.save('Pre_grx_%s' %name, G1_tf_gr[:,0,:,:])
    np.save('Pos_grx_%s' %name, G2_tf_gr[:,0,:,:])
    np.save('Pre_gry_%s' %name, G1_tf_gr[:,1,:,:])
    np.save('Pos_gry_%s' %name, G2_tf_gr[:,1,:,:])
    
    # Average over subjects and Normalization
    G1_x2y_avg = (np.mean(G1_x2y, axis=0)) 
    G2_x2y_avg = (np.mean(G2_x2y, axis=0)) 
    G1_y2x_avg = (np.mean(G1_y2x, axis=0)) 
    G2_y2x_avg = (np.mean(G2_y2x, axis=0)) 
    G1_grx_favg = sp.stats.zscore((np.mean(G1_tf_gr[:,0,:,:],0)) / (np.mean(G1_tf_gr[:,0,:,:],0)).max())
    G2_grx_favg = sp.stats.zscore((np.mean(G2_tf_gr[:,0,:,:],0)) / (np.mean(G2_tf_gr[:,0,:,:],0)).max())
    G1_gry_favg = sp.stats.zscore((np.mean(G1_tf_gr[:,1,:,:],0)) / (np.mean(G1_tf_gr[:,1,:,:],0)).max())
    G2_gry_favg = sp.stats.zscore((np.mean(G2_tf_gr[:,1,:,:],0)) / (np.mean(G2_tf_gr[:,1,:,:],0)).max()) 
    
    ## Subtraction of spectrums
    #G_xsub = G2_gr_favg[0,:,:] - G1_gr_favg[0,:,:]
    #G_ysub = G2_gr_favg[1,:,:] - G1_gr_favg[1,:,:]
    #print(G_xsub.shape)
    #print(G_ysub.shape)
    
    # Plotting Granger over Time

    fig1 = plt.figure(figsize=(15.0, 13.0))

    ax1 = fig1.add_subplot(2,1,1)
    stdev1 = np.std(G1_x2y_avg.T[:,0])
    plt.plot(times2save[0,:], G1_x2y_avg.T[:,0], label='v5 to v1')
    plt.tick_params(labelsize=20)
    plt.fill_between(times2save[0,:], G1_x2y_avg.T[:,0]+stdev1, G1_x2y_avg.T[:,0]-stdev1, alpha=.1)
    stdev2 = np.std(G1_y2x_avg.T[:,0])
    plt.plot(times2save[0,:], G1_y2x_avg.T[:,0], label='v1 to v5')
    plt.tick_params(labelsize=20)
    plt.fill_between(times2save[0,:], G1_y2x_avg.T[:,0]+stdev2, G1_y2x_avg.T[:,0]-stdev2, alpha=.1)
    plt.ylim([0.0,0.01])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    ax1.set_title('Baseline')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('G-causality')

    ax2 = fig1.add_subplot(2,1,2)
    stdev3 = np.std(G2_x2y_avg.T[:,0])
    plt.plot(times2save[0,:], G2_x2y_avg.T[:,0], label='v5 to v1')
    plt.tick_params(labelsize=20)
    plt.fill_between(times2save[0,:], G2_x2y_avg.T[:,0]+stdev3, G2_x2y_avg.T[:,0]-stdev3, alpha=.1)
    stdev4 = np.std(G2_y2x_avg.T[:,0])
    plt.plot(times2save[0,:], G2_y2x_avg.T[:,0], label='v1 to v5')
    plt.tick_params(labelsize=20)
    plt.fill_between(times2save[0,:], G2_y2x_avg.T[:,0]+stdev4, G2_y2x_avg.T[:,0]-stdev4, alpha=.1)
    plt.ylim([0.0,0.01])
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    ax2.set_title('Post')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('G-causality')
    
    fig1.savefig('TGr_%s' %name)
    #plt.show()

    # Plotting Time-Freq Granger
       
    levels = MaxNLocator(nbins=40).tick_values(-4, 5)

    fig2 = plt.figure(figsize=(20.0, 15.0))
    ax1 = fig2.add_subplot(2,2,1)
    CS_1 = plt.contourf(times2save[0,:], frex, G1_grx_favg, levels=levels, norm=colors.Normalize(-4, 5))
    cbar_1 = fig2.colorbar(CS_1, ax=ax1)
    plt.tick_params(labelsize=20)
    ax1.set_title('v5 to v1 Pre', fontsize=20)  
    plt.ylabel('Frequency', fontsize=20)
    plt.xlabel('Time(s)', fontsize=20)

    ax2 = fig2.add_subplot(2,2,2)
    CS_2 = plt.contourf(times2save[0,:], frex, G2_grx_favg, levels=levels, norm=colors.Normalize(-4, 5))
    cbar_2 = fig2.colorbar(CS_2, ax=ax2)
    plt.tick_params(labelsize=20)
    ax2.set_title('v5 to v1 Pos', fontsize=20)  
    plt.ylabel('Frequency', fontsize=20)
    plt.xlabel('Time(s)', fontsize=20)
    
    ax3 = fig2.add_subplot(2,2,3)
    CS_3 = plt.contourf(times2save[0,:], frex, G1_gry_favg, levels=levels, norm=colors.Normalize(-4, 5))
    cbar_3 = fig2.colorbar(CS_3, ax=ax3)
    plt.tick_params(labelsize=20)
    ax3.set_title('v1 to v5 Pre', fontsize=20)  
    plt.ylabel('Frequency', fontsize=20)
    plt.xlabel('Time(s)', fontsize=20)

    ax4 = fig2.add_subplot(2,2,4)
    CS_4 = plt.contourf(times2save[0,:], frex, G2_gry_favg, levels=levels, norm=colors.Normalize(-4, 5))
    cbar_4 = fig2.colorbar(CS_4, ax=ax4)
    plt.tick_params(labelsize=20)
    ax4.set_title('v1 to v5 Pos', fontsize=20)  
    plt.ylabel('Frequency', fontsize=20)
    plt.xlabel('Time(s)', fontsize=20)
    
    fig2.savefig('FrGr_%s' %name)
