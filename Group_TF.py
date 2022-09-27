import numpy as np
import scipy
import mne
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mne.viz import plot_topomap

#matplotlib.use('TkAgg')

from TimeFreqAna import *

def Group_TF (G1, G2, type_tr1, type_tr2, chansx, chansy, name, mode):
# G1: list of paths corresponding to 1st group
# G2: list of paths corresponding to 2nd group
# type_tr: Correct='stim_cor' or Incorrect='stim_inc'
# chansx: Channels from Region 1 to be averaged. Used always in mode = power and in mode = phase if PLL is desired 
# chansy: Channels from Region 2 to be averaged and used to calculate the PLL
# name: name to save the output image
# mode: Either power, phase, connect_all or pow_spec_den analysis of the electrodes

    # Initialize Group TimeFreq /Phase Arrays

    G1_s = np.empty((len(G1),40,750))
    G2_s = np.empty((len(G2),40,750))
    G1_avg = np.empty((40,750))
    G2_avg = np.empty((40,750))
    G_sub = np.empty((40,750))
    
    G1_ph_x = np.empty((len(G1), 40, 750), dtype=complex) # Undefined number of trials for every subject
    G2_ph_x = np.empty((len(G2), 40, 750), dtype=complex) # Not possible to preallocate space
    G1_ph_y = np.empty((len(G1), 40, 750), dtype=complex) 
    G2_ph_y = np.empty((len(G2), 40, 750), dtype=complex) 
    G1_itpc_x = np.empty((len(G1), 40, 750))
    G2_itpc_x = np.empty((len(G2), 40, 750))
    G1_itpc_y = np.empty((len(G1), 40, 750))
    G2_itpc_y = np.empty((len(G2), 40, 750))
    ph_diff_G1 = np.empty([])
    ph_diff_G2 = np.empty([])
    G1_pha = np.empty((len(G1), len(chansx), 40, 750), dtype=complex)
    G2_pha = np.empty((len(G2), len(chansy), 40, 750), dtype=complex)
    pha_diff_G1 = np.empty((len(chansx), len(chansx), len(G1), 40, 750))
    pha_diff_G2 = np.empty((len(chansy), len(chansy), len(G2), 40, 750))
    PSD1 = np.empty((len(G1), len(chansx), 125))
    PSD2 = np.empty((len(G2), len(chansy), 125))
    G1_pwr = np.empty((len(G1), len(chansx), 40, 750))
    G2_pwr = np.empty((len(G2), len(chansy), 40, 750))

    ispc_time_G1 = np.empty((0,750))
    ispc_time_G2 = np.empty((0,750))
    ispc_freq_G1 = np.empty((0,40))
    ispc_freq_G2 = np.empty((0,40))
    ispc_alpha_G1 = np.empty((0,750))
    ispc_alpha_G2 = np.empty((0,750))
    ispc_gamma_G1 = np.empty((0,750))
    ispc_gamma_G2 = np.empty((0,750)) 
    ispc_tri_G1 = np.empty([]) 
    ispc_tri_G2 = np.empty([])
    ispc_ang_tri_G1 = np.empty([])
    ispc_ang_tri_G2 = np.empty([])
    
#     PLV_alp_avg_G1 = np.empty((len(chansx),len(chansx)))
#     PLV_gam_avg = np.empty((len(chansx),len(chansx)))
    
    # Frequency Parameters
    min_freq = 2
    max_freq = 42
    num_freq = 40
    frex = np.linspace(min_freq, max_freq, num_freq)
    
    # Plotting TimeFreq Spectrum
    if mode == 'power':

        # Looping over subjects
        for id1, sub1 in enumerate(G1):
            G1_s[id1,:,:], _, _ = TimeFreqAna(sub1, type_tr1, chansx)

        for id2, sub2 in enumerate(G2):
            G2_s[id2,:,:], _, _ = TimeFreqAna(sub2, type_tr2, chansx) 

        # Saving Data for stats
        np.save('Pre_%s' %name, G1_s)
        np.save('Pos_%s' %name, G2_s)
        
#         # Read already saved data...
#         # '../../../Documents/Electrodes_May19/G1/Power/
#         G1_s = np.load('Pre_%s.npy' %name)
#         G2_s = np.load('Pos_%s.npy' %name)
            
        # Average over subjects
        G1_avg = np.mean(G1_s, axis=0)
        G2_avg = np.mean(G2_s, axis=0)

        # Subtraction of spectrums
        G_sub = G2_avg - G1_avg
        
        # Plotting
        
#         levels = MaxNLocator(nbins=40).tick_values(-4.25, 2.25)#tick_values(G1_avg.min(), G1_avg.max())
#         levels1 = MaxNLocator(nbins=40).tick_values(-1.25, 1.25)#tick_values(G1_avg.min(), G1_avg.max())
        vmin = -5
        vmax = 7
        levels = MaxNLocator(nbins=40).tick_values(vmin, vmax)
        levels1 = MaxNLocator(nbins=40).tick_values(-1.25, 1.25)

        fig1 = plt.figure(figsize=(15.0, 13.0))

        ax1 = fig1.add_subplot(3,1,1)
        CS_1 = plt.contourf(G1[1].times[350:550], frex, G1_avg[:,350:550], cmap='RdBu_r', levels=levels, norm=colors.Normalize(vmin=vmin, vmax=vmax), extend='both')
        cbar_1 = fig1.colorbar(CS_1, ax=ax1)
        ax1.set_title('Pre')
        #ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Frequency (Hz)')

        ax2 = fig1.add_subplot(3,1,2)
        CS_2 = plt.contourf(G1[1].times[350:550], frex, G2_avg[:,350:550], cmap='RdBu_r', levels=levels, norm=colors.Normalize(vmin=vmin, vmax=vmax), extend='both')
        cbar_2 = fig1.colorbar(CS_2, ax=ax2)
        ax2.set_title('Post')
        #ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')

        ax3 = fig1.add_subplot(3,1,3)
        CS_3 = plt.contourf(G1[1].times[350:550], frex, G_sub[:,350:550], cmap='RdBu_r', levels=levels1, norm=colors.Normalize(vmin=-1.25, vmax=1.25), extend='both')
        cbar = fig1.colorbar(CS_3, ax=ax3)
        ax3.set_title('Difference Post - Pre')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Frequency (Hz)')

        fig1.savefig('%s' %name)
        #plt.show()
        
    # Plotting Phase ITPC & ISPC
    elif mode == 'phase':
        
        # Looping over subjects
        for id1, sub1 in enumerate(G1):
            _, G1_ph_x[id1,:,:], G1_itpc_x[id1,:,:] = TimeFreqAna(sub1, type_tr1, chansx)
            _, G1_ph_y[id1,:,:], G1_itpc_y[id1,:,:] = TimeFreqAna(sub1, type_tr1, chansy)

        for id2, sub2 in enumerate(G2):
            _, G2_ph_x[id2,:,:], G2_itpc_x[id1,:,:] = TimeFreqAna(sub2, type_tr2, chansx)
            _, G2_ph_y[id2,:,:], G2_itpc_y[id1,:,:] = TimeFreqAna(sub2, type_tr2, chansy)
            
        # Phase Difference between ROIs --> x vs. y for both blocks G1 & G2
        ph_diff_G1 = np.angle(G1_ph_x[:,:,:]) - np.angle(G1_ph_y[:,:,:])
        ph_diff_G2 = np.angle(G2_ph_x[:,:,:]) - np.angle(G2_ph_y[:,:,:])

        ispc_t_G1 = np.exp(1j*ph_diff_G1) # subs x frex x time-points
        ispc_t_G2 = np.exp(1j*ph_diff_G2)
        ispc_f_G1 = np.exp(1j*ph_diff_G1) 
        ispc_f_G2 = np.exp(1j*ph_diff_G2)
        
        # Saving Data for stats
        np.save('Pre_t_%s' %name, ispc_t_G1)
        np.save('Pos_t_%s' %name, ispc_t_G2)
        np.save('Pre_f_%s' %name, ispc_f_G1)
        np.save('Pos_f_%s' %name, ispc_f_G2)
        
        ispc_time_G1 = abs(np.mean(np.mean(np.exp(1j*ph_diff_G1),0),0)) # subs x frex x time-points
        ispc_time_G2 = abs(np.mean(np.mean(np.exp(1j*ph_diff_G2),0),0))
        ispc_freq_G1 = abs(np.mean(np.mean(np.exp(1j*ph_diff_G1),0),1))
        ispc_freq_G2 = abs(np.mean(np.mean(np.exp(1j*ph_diff_G2),0),1))

        ispc_alpha_G1 = ((np.angle(np.mean(np.mean(np.exp(1j*ph_diff_G1),0)[5:11],0))) +2*np.pi) % (2*np.pi)
        ispc_alpha_G2 = ((np.angle(np.mean(np.mean(np.exp(1j*ph_diff_G2),0)[5:11],0))) +2*np.pi) % (2*np.pi)
        ispc_gamma_G1 = ((np.angle(np.mean(np.mean(np.exp(1j*ph_diff_G1),0)[28:40],0))) +2*np.pi) % (2*np.pi)
        ispc_gamma_G2 = ((np.angle(np.mean(np.mean(np.exp(1j*ph_diff_G2),0)[28:40],0))) +2*np.pi) % (2*np.pi)

        ispc_tri_G1 = abs(np.mean(np.mean(np.exp(1j*ph_diff_G1[:,:,372:378]),2),1)) # evaluated at the stimulus onset
        ispc_tri_G2 = abs(np.mean(np.mean(np.exp(1j*ph_diff_G2[:,:,372:378]),2),1)) 
        ispc_ang_tri_G1 = np.angle(np.mean(np.mean(np.exp(1j*ph_diff_G1[:,:,372:378]),2),1)) 
        ispc_ang_tri_G2 = np.angle(np.mean(np.mean(np.exp(1j*ph_diff_G2[:,:,372:378]),2),1)) 
        

        # Phase Difference between G1 & G2
        #ispc_alpha = ispc_alpha_G2 - ispc_alpha_G1 
        #ispc_gamma = ispc_gamma_G2 - ispc_gamma_G1 
        
        # ITPC averaging over subjects and normalizing
        G1_itpc_x = (np.mean(G1_itpc_x, 0))/ (np.mean(G1_itpc_x, 0)).max()
        G2_itpc_x = (np.mean(G2_itpc_x, 0))/ (np.mean(G2_itpc_x, 0)).max()
        G1_itpc_y = (np.mean(G1_itpc_y, 0))/ (np.mean(G1_itpc_y, 0)).max()
        G2_itpc_y = (np.mean(G2_itpc_y, 0))/ (np.mean(G2_itpc_y, 0)).max()
        itpc = [G1_itpc_x, G1_itpc_y, G2_itpc_x, G2_itpc_y]
        
        # Plotting
        
        fig1 = plt.figure(figsize=(13.0, 15.0))
        for gr_idx, gr in enumerate(itpc):
            #levels = MaxNLocator(nbins=40).tick_values(0, 1)
            ax = fig1.add_subplot(2,2,gr_idx+1)
            CS = plt.contourf(G1[1].times[350:550], frex, gr[:,350:550])#, levels=levels)#, norm=colors.Normalize(vmin=0, vmax=1))
            cbar = fig1.colorbar(CS)
            plt.tick_params(labelsize=20)
            if gr_idx == 0:
                ax.set_title('Pre Parietal', fontsize=20)  
            elif gr_idx == 1:
                ax.set_title('Pre Occipital', fontsize=20) 
            elif gr_idx == 2:
                ax.set_title('Pos Parietal', fontsize=20)
            elif gr_idx == 3:
                ax.set_title('Pos Occipital', fontsize=20)
            plt.ylabel('Single Phase Synchrony', fontsize=20)
            plt.xlabel('Time(s)', fontsize=20)
        fig1.savefig('Single_%s' %name)

        fig2 = plt.figure(figsize=(13.0, 15.0))
        ax1 = fig2.add_subplot(2,1,1)
        stdev1 = np.std(ispc_time_G1)
        plt.plot(G1[1].times[350:550], ispc_time_G1[350:550], label='Pre')
        plt.tick_params(labelsize=20)
        plt.fill_between(G1[1].times[350:550], ispc_time_G1[350:550]+stdev1, ispc_time_G1[350:550]-stdev1, alpha=.1)
        stdev2 = np.std(ispc_time_G2)
        plt.plot(G2[1].times[350:550], ispc_time_G2[350:550], label='Pos') 
        plt.tick_params(labelsize=20)
        plt.fill_between(G2[1].times[350:550], ispc_time_G2[350:550]+stdev2, ispc_time_G2[350:550]-stdev2, alpha=.1)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.ylabel('Pair Phase Synchrony - PLV', fontsize=20)
        plt.xlabel('Time(s)', fontsize=20)
        plt.ylim([0,0.4])
        ax2 = fig2.add_subplot(2,1,2)
        stdev3 = np.std(ispc_freq_G1)
        plt.plot(frex, ispc_freq_G1, label='Pre')
        plt.tick_params(labelsize=20)
        plt.fill_between(frex, ispc_freq_G1+stdev3, ispc_freq_G1-stdev3, alpha=.1)
        stdev4 = np.std(ispc_freq_G2)
        plt.plot(frex, ispc_freq_G2, label='Pos')
        plt.tick_params(labelsize=20)
        plt.fill_between(frex, ispc_freq_G2+stdev4, ispc_freq_G2-stdev4, alpha=.1)
        plt.tick_params(labelsize=20)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.ylabel('Pair Phase Synchrony - PLV', fontsize=20)
        plt.xlabel('Frequency(Hz)', fontsize=20)
        plt.ylim([0,0.4])    
        fig2.savefig('Pair_%s' %name)

        fig3 = plt.figure(figsize=(13.0, 15.0))
        ax3 = fig3.add_subplot(2,1,1)
        stdev5 = np.std(ispc_alpha_G1)
        plt.plot(G1[1].times[350:550], ispc_alpha_G1[350:550], label='Pre')
        plt.fill_between(G1[1].times[350:550], ispc_alpha_G1[350:550]+stdev5, ispc_alpha_G1[350:550]-stdev5, alpha=.1)
        stdev6 = np.std(ispc_alpha_G2)
        plt.plot(G2[1].times[350:550], ispc_alpha_G2[350:550], label='Pos')
        plt.fill_between(G1[1].times[350:550], ispc_alpha_G2[350:550]+stdev6, ispc_alpha_G2[350:550]-stdev6, alpha=.1)
        plt.tick_params(labelsize=20)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        ax3.set_title('Alpha Band', fontsize=20)  
        plt.ylabel('Phase (Rad)', fontsize=20)
        plt.xlabel('Time(s)', fontsize=20)
        plt.ylim(0, 2*np.pi)
        ax4 = fig3.add_subplot(2,1,2)
        stdev7 = np.std(ispc_gamma_G1)
        plt.plot(G1[1].times[350:550], ispc_gamma_G1[350:550], label='Pre')
        plt.fill_between(G1[1].times[350:550], ispc_gamma_G1[350:550]+stdev7, ispc_gamma_G1[350:550]-stdev7, alpha=.1)
        stdev8 = np.std(ispc_gamma_G2)
        plt.plot(G2[1].times[350:550], ispc_gamma_G2[350:550], label='Pos')
        plt.fill_between(G1[1].times[350:550], ispc_gamma_G2[350:550]+stdev8, ispc_gamma_G2[350:550]-stdev8, alpha=.1)
        plt.tick_params(labelsize=20)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        ax4.set_title('Gamma Band', fontsize=20)  
        plt.ylabel('Phase (Rad)', fontsize=20)
        plt.xlabel('Time(s)', fontsize=20)
        plt.ylim(0, 2*np.pi)
        fig3.savefig('Alpha_Gamma_%s' %name)
        
        fig4 = plt.figure(figsize=(6.0, 15.0))
        ax5 = fig4.add_subplot(2,1,1, projection='polar')
        ax5.set_title('Baseline', fontsize=20)
        ph_G1=np.empty([2,15])
        ma_G1=np.empty([2,15])
        for c,_ in enumerate(G1):
            ph_G1[:,c]=np.array([0, ispc_ang_tri_G1[c]])
            ma_G1[:,c]=np.array([0, ispc_tri_G1[c]])
        plt.polar(ph_G1, ma_G1, marker='o')
        plt.tick_params(labelsize=20)
        ax6 = fig4.add_subplot(2,1,2, projection='polar')
        ax6.set_title('Pos', fontsize=20) 
        ph_G2=np.empty([2,15])
        ma_G2=np.empty([2,15])
        for c,_ in enumerate(G2):
            ph_G2[:,c]=np.array([0, ispc_ang_tri_G2[c]])
            ma_G2[:,c]=np.array([0, ispc_tri_G2[c]])
        plt.polar(ph_G2, ma_G2, marker='o')        
        #plt.scatter((0, ispc_ang_tri_G2), (0, ispc_tri_G2), c=colors, marker='o')
        plt.tick_params(labelsize=20)
        fig4.savefig('Polar_Onset_%s' %name)
        
        fig5 = plt.figure(figsize=(6.0, 15.0))
        ax7 = fig5.add_subplot(2,1,1, projection='polar')
        ax7.set_title('Alpha', fontsize=20)
#         ph_G1=np.empty([2,15])
#         ma_G1=np.empty([2,15])
#         for c,_ in enumerate(G1):
#             ph_G1[:,c]=np.array([0, ispc_ang_tri_G1[c]])
#             ma_G1[:,c]=np.array([0, ispc_tri_G1[c]])
        plt.polar(G1[1].times[350:550], ispc_alpha_G1[350:550], label='Pre', marker='o')
        plt.polar(G2[1].times[350:550], ispc_alpha_G2[350:550], label='Pos', marker='o')
        plt.tick_params(labelsize=20)
        ax8 = fig5.add_subplot(2,1,2, projection='polar')
        ax8.set_title('Gamma', fontsize=20) 
#         ph_G2=np.empty([2,15])
#         ma_G2=np.empty([2,15])
#         for c,_ in enumerate(G2):
#             ph_G2[:,c]=np.array([0, ispc_ang_tri_G2[c]])
#             ma_G2[:,c]=np.array([0, ispc_tri_G2[c]])
        plt.polar(G1[1].times[350:550], ispc_gamma_G1[350:550], label='Pre', marker='o') 
        plt.polar(G2[1].times[350:550], ispc_gamma_G2[350:550], label='Pos', marker='o')
        #plt.scatter((0, ispc_ang_tri_G2), (0, ispc_tri_G2), c=colors, marker='o')
        plt.tick_params(labelsize=20)
        fig5.savefig('Polar_AlphaGamma_%s' %name)
        
    # Plotting TimeFreq Spectrum
    elif mode == 'connect_all':
               
        # Get phase data for every electrode and every subject
        for id_sub, sub in enumerate(G1):
             for id_ch, ch in enumerate(chansx):
                _, G1_pha[id_sub,id_ch,:,:], _ = TimeFreqAna(sub, type_tr1, [ch,]) #Pass channel as single-value tuple
                
        for id_sub, sub in enumerate(G2):
             for id_ch, ch in enumerate(chansy):
                _, G2_pha[id_sub,id_ch,:,:], _ = TimeFreqAna(sub, type_tr2, [ch,]) #Pass channel as single-value tuple
        
        # Phase difference among all electrodes
        for i in range(len(chansx)):
            for j in range(len(chansx)):
                pha_diff_G1[i,j,:,:,:] = np.angle(G1_pha[:,i,:,:]) - np.angle(G1_pha[:,j,:,:]) # ch x ch x subs x frex x time-points
                
        for i in range(len(chansy)):
            for j in range(len(chansy)):
                pha_diff_G2[i,j,:,:,:] = np.angle(G2_pha[:,i,:,:]) - np.angle(G2_pha[:,j,:,:]) # ch x ch x subs x frex x time-points
                
        # Average over subjects
        G1_avg = np.mean(np.exp(1j*pha_diff_G1),2)
        G2_avg = np.mean(np.exp(1j*pha_diff_G2),2)
        
        # Calculate PLV during stimulus presentation 
        PLV_alp_avg_G1 = abs(np.mean(np.mean(G1_avg[:,:,5:11,:],2)[:,:,375:500],2)) # ch x ch
        PLV_gam_avg_G1 = abs(np.mean(np.mean(G1_avg[:,:,28:40,:],2)[:,:,375:500],2)) # ch x ch
        PLV_alp_avg_G2 = abs(np.mean(np.mean(G2_avg[:,:,5:11,:],2)[:,:,375:500],2)) # ch x ch
        PLV_gam_avg_G2 = abs(np.mean(np.mean(G2_avg[:,:,28:40,:],2)[:,:,375:500],2)) # ch x ch
        np.save('Alpha_Bsl_%s' %name, PLV_alp_avg_G1)
        np.save('Gamma_Bsl_%s' %name, PLV_gam_avg_G1)
        np.save('Alpha_P1_%s' %name, PLV_alp_avg_G2)
        np.save('Gamma_P1_%s' %name, PLV_gam_avg_G2)
        
        # Plotting contourf
        chx = np.linspace(1, len(chansx), len(chansx))
        
        levels = MaxNLocator(nbins=40).tick_values(0, 1)#tick_values(G1_avg.min(), G1_avg.max())

        fig6 = plt.figure(figsize=(15.0, 13.0))

        ax1 = fig6.add_subplot(2,2,1)
        CS_1 = plt.contourf(chx, chx, PLV_alp_avg_G1, levels=levels, norm=colors.Normalize(vmin=0, vmax=1))
        cbar_1 = fig6.colorbar(CS_1, ax=ax1)
        ax1.set_title('Alpha Baseline')
        ax1.set_xlabel('Channels')
        ax1.set_ylabel('Channels')
        
        ax2 = fig6.add_subplot(2,2,2)
        CS_2 = plt.contourf(chx, chx, PLV_gam_avg_G1, levels=levels, norm=colors.Normalize(vmin=0, vmax=1))
        cbar_2 = fig6.colorbar(CS_2, ax=ax2)
        ax2.set_title('Gamma Baseline')
        ax2.set_xlabel('Channels')
        ax2.set_ylabel('Channels')
        
        ax3 = fig6.add_subplot(2,2,3)
        CS_3 = plt.contourf(chx, chx, PLV_alp_avg_G2, levels=levels, norm=colors.Normalize(vmin=0, vmax=1))
        cbar_3 = fig6.colorbar(CS_3, ax=ax3)
        ax3.set_title('Alpha Post')
        ax3.set_xlabel('Channels')
        ax3.set_ylabel('Channels')
        
        ax4 = fig6.add_subplot(2,2,4)
        CS_4 = plt.contourf(chx, chx, PLV_gam_avg_G2, levels=levels, norm=colors.Normalize(vmin=0, vmax=1))
        cbar_4 = fig6.colorbar(CS_4, ax=ax4)
        ax4.set_title('Gamma Post')
        ax4.set_xlabel('Channels')
        ax4.set_ylabel('Channels')
        
        fig6.savefig('PLV_%s' %name)
        plt.show()
        
    # Plotting TimeFreq Spectrum
    elif mode == 'pow_spec_den':
        
        def find_ch_index(sub,ch_interest):
            
            chan2use = ch_interest
            channels = sub.info['ch_names']
            count_ch = 0
            for ch in chan2use:
                count_ch = count_ch + channels.count(ch)
            ch_index = np.zeros([count_ch,1], dtype=int)

            i = 0
            for idx, chan in enumerate(channels):
                for ch, _ in enumerate(chan2use):
                    if chan == chan2use[ch]:
                        ch_index[i,0] = int(idx)
                        i=i+1
                    else:
                        i=i         
#             print(ch_index)
            return ch_index
               
        # Looping over subjects
        for id1, sub1 in enumerate(G1):
            
            data1 = sub1['stim_cor'].get_data()
            ch_index1 = find_ch_index(sub1,chansx)
                      
            for ch1 in range(ch_index1.shape[0]):
#                 print(data1.shape)
                f, PSD1[id1, ch1, :] = signal.welch(np.reshape(data1[:,ch_index1[ch1,0],:], (1, data1.shape[0]*data1.shape[2])),
                                                    sub1.info['sfreq'], nperseg=sub1.info['sfreq']-1)
            np.save('Pre_PSD_%s' %name, PSD1)
            print(id1)

        for id2, sub2 in enumerate(G2):
            
            data2 = sub2['stim_cor'].get_data()
            ch_index2 = find_ch_index(sub2,chansy)
                      
            for ch2 in range(ch_index2.shape[0]):
                f, PSD2[id2, ch2, :] = signal.welch(np.reshape(data2[:,ch_index1[ch2,0],:], (1, data2.shape[0]*data2.shape[2])), 
                                                    sub2.info['sfreq'], nperseg=sub1.info['sfreq']-1)   
            np.save('Pos_PSD_%s' %name, PSD2)
            print(id2+15)
            
#                 # Read already saved data...
#         # '../../../Documents/Electrodes_May19/G1/Power/
#         PSD1 = np.load('Pre_PSD_%s.npy' %name)
#         PSD2 = np.load('Pos_PSD_%s.npy' %name)
               
        # Averaging over subjects
        psd1 = np.mean(PSD1, 0)
        psd2 = np.mean(PSD2, 0)
        
        # Plotting PSD channels
        fig7 = plt.figure(figsize=(15, 7))
            ## Toggle commenting when plotting individual Channels PSD
#         for chan, ch_name in enumerate(chansx): 
#             print(chan, ch_name)
#             ax7 = fig7.add_subplot(1,2,chan+1)
#             f = np.arange(psd1.shape[-1]) * (250/255)
#             plt.plot(f, psd1[chan,:], label='Pre tACS')
#             plt.plot(f, psd2[chan,:], label='Pos tACS')
#             plt.title(ch_name, fontsize=20)
        plt.plot(f, np.mean(psd1, 0), label='Pre tACS')
        plt.plot(f, np.mean(psd2, 0), label='Pos tACS')
        plt.title('PSD', fontsize=20)
            ## Move lines below inside the loop when looking at individual Ch
        plt.legend(loc='upper right', fontsize='large')
        plt.xlim([0, 45])
        plt.xlabel('frequency [Hz]', fontsize=20)
        plt.ylabel('PSD [V**2/Hz]', fontsize=20)
        plt.tick_params(labelsize=20)
        fig7.savefig('Psd_%s' %name)
            ##
        plt.show()
        
        # Plotting topo
        montage = mne.channels.read_montage(kind='easycap_64')
        montage.selection = montage.selection[:64]
        info = mne.create_info(montage.ch_names[:64], 250, 'eeg', montage=montage)
               
        fig8, ax_topo = plt.subplots(2, 3, figsize=(15, 13))
#         print(ax_topo[0])
        plot_topomap(np.mean(psd1[:,8:14],1), pos=info, axes=ax_topo[0,0], cmap='RdBu_r',
                            vmin=0, vmax=np.max, show=False)
        ax_topo[0,0].set_title('Alpha Pre Stim')
        plot_topomap(np.mean(psd1[:,15:31],1), pos=info, axes=ax_topo[0,1], cmap='RdBu_r',
                            vmin=0, vmax=np.max, show=False)
        ax_topo[0,1].set_title('Beta Pre Stim')
        plot_topomap(np.mean(psd1[:,31:50],1), pos=info, axes=ax_topo[0,2], cmap='RdBu_r',
                            vmin=0, vmax=np.max, show=False)
        ax_topo[0,2].set_title('Gamma Pre Stim')
        plot_topomap(np.mean(psd2[:,8:14],1), pos=info, axes=ax_topo[1,0], cmap='RdBu_r',
                            vmin=0, vmax=np.max, show=False)
        ax_topo[1,0].set_title('Alpha Pos Stim')
        plot_topomap(np.mean(psd2[:,15:31],1), pos=info, axes=ax_topo[1,1], cmap='RdBu_r',
                            vmin=0, vmax=np.max, show=False)
        ax_topo[1,1].set_title('Beta Pre Stim')
        plot_topomap(np.mean(psd2[:,31:50],1), pos=info, axes=ax_topo[1,2], cmap='RdBu_r',
                            vmin=0, vmax=np.max, show=False)
        ax_topo[1,2].set_title('Gamma Pre Stim')
        fig8.savefig('Pwr_topo_%s' %name)
        plt.show()
        
    # Full brain
    elif mode == 'brain_topo_tf':
               
        # Get power data for every electrode and every subject
#         for id_sub1, sub1 in enumerate(G1):
#             print(id_sub1)
#             for id_ch1, ch1 in enumerate(chansx):
#                 G1_pwr[id_sub1,id_ch1,:,:], _, _ = TimeFreqAna(sub1, type_tr1, [ch1,]) #Pass channel as single-value tuple
#         np.save('%s_Bsl' %name, G1_pwr)
                
#         for id_sub2, sub2 in enumerate(G2):
#             print(id_sub2+15)
#             for id_ch2, ch2 in enumerate(chansy):
#                 G2_pwr[id_sub2,id_ch2,:,:], _, _ = TimeFreqAna(sub2, type_tr2, [ch2,]) #Pass channel as single-value tuple
#         np.save('%s_Pos1' %name, G2_pwr)
        
#         # Read already saved data...
        G1_pwr = np.load('%s_Bsl.npy' %name) # Subs x Ch x Frex x Timepoints
        G2_pwr = np.load('%s_Pos1.npy' %name)

        # Average over subjects and time
        G1_avg = np.mean(G1_pwr,axis=(0, 3))
        G2_avg = np.mean(G2_pwr,axis=(0, 3))
        G1_avg[62,:] = 10
        G2_avg[62,:] = 10
               
        # Plotting topoplot Alpha Band
        montage = mne.channels.read_montage(kind='easycap_64')
        montage.selection = montage.selection[:64]
        info = mne.create_info(montage.ch_names[:64], 250, 'eeg', montage=montage)
               
        fig8, ax_topo = plt.subplots(2, 1, figsize=(15, 13))
        print(ax_topo[0])
        plot_topomap(np.mean(G1_avg[:,5:11],1), pos=info, axes=ax_topo[0], cmap='RdBu_r',
                            vmin=0, vmax=np.max, show=False)
        ax_topo[0].set_title('Alpha Pre Stim')
        print(ax_topo[1])
        plot_topomap(np.mean(G2_avg[:,5:11],1), pos=info, axes=ax_topo[1], cmap='RdBu_r',
                            vmin=0, vmax=np.max, show=False)
        ax_topo[1].set_title('Alpha Pos Stim')
        
        fig8.savefig('Pwr_topo_%s' %name)
        plt.show()
        


        
        