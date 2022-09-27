import numpy as np
import scipy
import mne
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import mayavi
from mayavi import mlab

from SourcesReco_Bursts import *

def Group_Src_Bursts (G1, G2, type_tr1, type_tr2, name1, name2, mode):
# G1: list of paths corresponding to 1st group
# G2: list of paths corresponding to 2nd group
# type_tr: Correct='stim_cor' or Incorrect='stim_inc'
# name1: name to save the output image
# name2: name to save the output image
# mode: Either band_power, tf_power, phase or cross_freq analysis of the electrodes

    # Initialize Group TimeFreq /Phase Arrays

    G1_con = np.empty([2,2,44,1000,len(G1)])
    G2_con = np.empty([2,2,44,1000,len(G1)])
    G1_cau = np.empty([2,2,4,1000,30])
    G2_cau = np.empty([2,2,4,1000,30])
    G1_tf_pwr = np.empty([7,44,1000])
    G2_tf_pwr = np.empty([7,44,1000])
    G1_cf = np.empty([5,6,2,15])
    G2_cf = np.empty([5,6,2,30])
    G1_pwr = np.empty([len(G1),44,1000])
    G2_pwr = np.empty([len(G1),44,1000])
    G1_wave = np.empty([len(G1),2,44,1000], dtype=complex)
    G2_wave = np.empty([len(G1),2,44,1000], dtype=complex)
    G1_twspeed = np.empty([len(G1),44,1000])
    G2_twspeed = np.empty([len(G1),44,1000])
    G1_data_alpha = np.empty([2,750,15], dtype=complex)
    G2_data_alpha = np.empty([2,750,15], dtype=complex)
    G1_data_gamma = np.empty([2,750,15], dtype=complex)
    G2_data_gamma = np.empty([2,750,15], dtype=complex)
    
    # Times of interest
    min_time = -1.5#-1.5
    max_time = 2.5#1.5
    num_time = 1000#750
    timex = np.linspace(min_time, max_time, num_time)
    
    # Frequency Parameters
    min_freq = 1#2
    max_freq = 99#42
    num_freq = 44#40
    frex = np.linspace(min_freq, max_freq, num_freq)  


    # Plotting  TF Power (ROI sources)
    if mode == 'tf_power':

#          # Looping over subjects to get data --> To comment when data has already been computed
        for no, (sub1, sub2) in enumerate(zip(G1, G2)):
            SourcesReco_Bursts(sub1, type_tr1, name1, no+1, mode)
            SourcesReco_Bursts(sub2, type_tr2, name2, no+1, mode) 
            
        # Reading subject's data in Src.Space
        for a in range(len(G1)):
            G1_tf_pwr[a,:,:] = np.load('Pwr_%s_%s.npy' %(name1, a+1))
            G2_tf_pwr[a,:,:] = np.load('Pwr_%s_%s.npy' %(name1, a+1) %(name2, a+1))
        
        # Averaging power over subjects 
        G1_avg = np.mean(G1_tf_pwr, axis=0)
        G2_avg = np.mean(G2_tf_pwr, axis=0)  
        
        # Subtraction of spectrums
        G_sub = G2_avg - G1_avg
        
        # Plotting
        vmin = -5
        vmax = 7
        levels = MaxNLocator(nbins=40).tick_values(vmin, vmax)
        levels1 = MaxNLocator(nbins=40).tick_values(-1.25, 1.25)
        
#         plt.imshow(20 * G_sub,
#                    extent=[timex[300], timex[500], frex[0], frex[-1]],
#                    aspect='auto', origin='lower', vmin=0., vmax=30., cmap='RdBu_r')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Frequency (Hz)')
#         #plt.title('Power (%s)' % title)
#         plt.colorbar()
#         plt.show()

        fig1 = plt.figure(figsize=(15.0, 13.0))

        ax1 = fig1.add_subplot(3,1,1)
        CS_1 = plt.contourf(timex, frex, G1_avg, cmap='RdBu_r', levels=levels, norm=colors.Normalize(vmin=vmin, vmax=vmax), extend='both')
        cbar_1 = fig1.colorbar(CS_1, ax=ax1)
        ax1.set_title('Pre')
        #ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Frequency (Hz)')

        ax2 = fig1.add_subplot(3,1,2)
        CS_2 = plt.contourf(timex[350:550], frex, G2_avg[:,350:550], cmap='RdBu_r', levels=levels, norm=colors.Normalize(vmin=vmin, vmax=vmax), extend='both')
        cbar_2 = fig1.colorbar(CS_2, ax=ax2)
        ax2.set_title('Post')
        #ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')

        ax3 = fig1.add_subplot(3,1,3)
        CS_3 = plt.contourf(timex[350:550], frex, G_sub[:,350:550], cmap='RdBu_r', levels=levels1, norm=colors.Normalize(vmin=-1.25, vmax=1.25), extend='both')
        cbar = fig1.colorbar(CS_3, ax=ax3)
        ax3.set_title('Difference Post - Pre')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Frequency (Hz)')

        fig1.savefig('Src_TF_%s_%s' %(name1, name2))
            
    # Plotting Connectivity Graphs
    if mode == 'connect':
        
        # Looping over subjects to get data
        for no, (sub1, sub2) in enumerate(zip(G1, G2)):
            SourcesReco_Bursts(sub1, type_tr1, name1, no+1, mode)
            SourcesReco_Bursts(sub2, type_tr2, name2, no+1, mode) 
            
        # Reading subject's data in Src.Space
        print(len(G1), len(G2))
        for a in range(len(G1)):
            G1_con[:,:,:,:,a] = np.load('WPLI_%s_%s.npy' %(name1, a+1))
            G2_con[:,:,:,:,a] = np.load('WPLI_%s_%s.npy' %(name2, a+1))
#         for a in range(len(G2)):
#             if a <= 14:
#                 G2_cau[:,:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/Psi/PSI_%s_BSham_%s.npy' %(name1, a+1))   #Sham InP
#             elif a > 14:
#                 G2_cau[:,:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/Psi/PSI_%s_BSham_%s.npy' %(name2, a-14))  #Sham AnP

        # Total Verum vs. Total Sham
#             if a <= 14:
#         # New Vectors to be filled when computing Phase Slope Index (4 Averaged Frex Bands instead of 40 Frex bins)
#                 G1_cau[:,:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/Psi/PSI_%s_Bursts_%s.npy' %(name1, a+1))
#                 G2_cau[:,:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/Psi/PSI_%s_BSham_%s.npy' %(name1, a+1))
#             elif a > 14:
#                 G1_cau[:,:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/Psi/PSI_%s_Bursts_%s.npy' %(name2, a-14))
#                 G2_cau[:,:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/Psi/PSI_%s_BSham_%s.npy' %(name2, a-14))
        
        print(G1_con.shape) # ([2,2,40,1000,15]) #Connectivity between 2 labels it is always found in the position [1,0,:,:,:]
#         print(G1_cau.shape, G2_cau.shape) # ([2,2,4,1000,15]) # Upper positive matrix is filled PSI
#         print(G1_con.dtype)
            
        # Averaging Conn over subjects 
        G1_con_avg = np.mean(G1_con, axis=4)
        G2_con_avg = np.mean(G2_con, axis=4)
        # Change for PSI - Average over subjects
#         G1_con_avg = np.mean(G1_cau, axis=4)
#         G2_con_avg = np.mean(G2_cau, axis=4)
        
                        
        def plot_tf (G1, G2, timex, frex, name):
            
            diffmap = np.subtract(G1,G2)
           
            levels = MaxNLocator(nbins=40).tick_values(0, 0.25)
            levels1 = MaxNLocator(nbins=40).tick_values(-0.2, 0.2)
            
            # To use when plotting PSI -> Change variables in contourf : Otherwise change for 'frex'
            frex_cau = np.array([2,7,30,45])

            fig1 = plt.figure(figsize=(15, 10))#figsize=(27.0, 3.0))

            ax1 = fig1.add_subplot(3,1,1)
            CS_1 = plt.contourf(timex[350:550], frex, G1[:,350:550], cmap='RdBu_r', levels=levels, extend='both')
            cbar_1 = fig1.colorbar(CS_1, ax=ax1)
            ax1.set_title('VERUM', fontsize=20)
#             ax1.set_xlabel('Time(s)', fontsize=20)
            ax1.set_ylabel('Freq.(Hz)', fontsize=20)

            ax2 = fig1.add_subplot(3,1,2)
            CS_2 = plt.contourf(timex[350:550], frex, G2[:,350:550], cmap='RdBu_r', levels=levels, extend='both')
            cbar_2 = fig1.colorbar(CS_2, ax=ax2)
            ax2.set_title('SHAM', fontsize=20)
#             ax2.set_xlabel('Time(s)', fontsize=20)
            ax2.set_ylabel('Freq.(Hz)', fontsize=20)
            
            ax3 = fig1.add_subplot(3,1,3)
            CS_3 = plt.contourf(timex[350:550], frex, diffmap[:,350:550], cmap='RdBu_r', levels=levels1, extend='both')
            cbar_3 = fig1.colorbar(CS_3, ax=ax3)
            ax3.set_title('Verum-Sham', fontsize=20)
            ax3.set_xlabel('Time(s)', fontsize=20)
            ax3.set_ylabel('Freq.(Hz)', fontsize=20)

            plt.show()
            fig1.savefig('%s' %name)
        
        for b in range(2):
            for c in range(b):
                plot_tf(G1_con_avg[b,c,:,:], G2_con_avg[b,c,:,:], timex, frex, 'WPLI_Src_%s_%s%s' %(name1, b,c))

    # Cross-Frequency analysis
    if mode == 'cross_freq':
        
        # Looping over subjects to get data
#         for no, (sub1, sub2) in enumerate(zip(G1, G2)):
#             SourcesReco_Bursts(sub1, type_tr1, name1, no+1, mode)
#             SourcesReco_Bursts(sub2, type_tr2, name2, no+1, mode) 
            
        # Reading subject's data in Src.Space
        print(len(G1), len(G2))
        for a in range(len(G1)):
            G1_cf[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_Bursts_%s.npy' %(name2, a+1))
#             G2_cf[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_%s.npy' %(name2, a+1))  
        for a in range(len(G2)):
            if a <= 14:
                G2_cf[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_BSham_%s.npy' %(name1, a+1))   #Sham InP
            elif a > 14:
                G2_cf[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_BSham_%s.npy' %(name2, a-14))  #Sham AnP
                
        # Total Verum vs. Total Sham
#         for a in range(len(G1)):
#             if a <= 14:
#                 G1_cf[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_Bursts_%s.npy' %(name1, a+1))   #Verum InP
#                 G2_cf[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_BSham_%s.npy' %(name1, a+1))    #Sham InP
#             elif a > 14:
#                 G1_cf[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_Bursts_%s.npy' %(name2, a-14))  #Verum AnP
#                 G2_cf[:,:,:,a] = np.load('../../../Documents/3rd_Results/Subjects/CrossFr/zPAC_V1V5_%s_BSham_%s.npy' %(name2, a-14))   #Sham AnP
        
        print(G1_cf.shape, G2_cf.shape) # ([5,6,2,15])
        
#         # Averaging CF over subjects 
        G1_V1pV5a_avg = np.mean(G1_cf[:,:,0,:], axis=2)
        G2_V1pV5a_avg = np.mean(G2_cf[:,:,0,:], axis=2)
        G1_V1aV5p_avg = np.mean(G1_cf[:,:,1,:], axis=2)
        G2_V1aV5p_avg = np.mean(G2_cf[:,:,1,:], axis=2)
        
        # Lo Frequency Parameters
        frex_lo = np.linspace(3, 15, 6)

        # Hi Frequency Parameters
        frex_hi = np.linspace(22, 42, 5)
        
        def plot_tf (G1, G2, x, y, name):
        
            diffmap = np.subtract(G1,G2)
#             diffmap = np.divide(G2,G1)

            levels = MaxNLocator(nbins=60).tick_values(-1, 1)
            levels1 = MaxNLocator(nbins=60).tick_values(-0.75, 0.75) #Difference
#             levels1 = MaxNLocator(nbins=60).tick_values(0.25, 3.0) #Ratio

            fig1 = plt.figure(figsize=(15.0,4.0))

            ax1 = fig1.add_subplot(1,3,1)
            CS_1 = plt.contourf(x, y, G1, cmap='RdBu_r', levels=levels, extend='both')
            cbar_1 = fig1.colorbar(CS_1, ax=ax1)
            ax1.set_title('VERUM')
            ax1.set_xlabel('Frequency for Phase(Hz)')
            ax1.set_ylabel('Freq. for Amplitude(Hz)')

            ax2 = fig1.add_subplot(1,3,2)
            CS_2 = plt.contourf(x, y, G2, cmap='RdBu_r', levels=levels, extend='both')
            cbar_2 = fig1.colorbar(CS_2, ax=ax2)
            ax2.set_title('SHAM')
            ax2.set_xlabel('Frequency for Phase(Hz)')
#             ax2.set_ylabel('Freq. for Amplitude(Hz)')

            ax3 = fig1.add_subplot(1,3,3)
            CS_3 = plt.contourf(x, y, diffmap, cmap='RdBu_r', extend='both', levels=levels)
            cbar = fig1.colorbar(CS_3, ax=ax3)
            ax3.set_title('Verum-Sham')
#             ax3.set_title('Ratio Pos/Pre')
            ax3.set_xlabel('Frequency for Phase(Hz)')
#             ax3.set_ylabel('Freq. for Amplitude(Hz)')

            plt.show()
            fig1.savefig('%s' %name)

        plot_tf(G1_V1pV5a_avg, G2_V1pV5a_avg, frex_lo, frex_hi, 'CF_V1pV5a_G2-Sh')
        plot_tf(G1_V1aV5p_avg, G2_V1aV5p_avg, frex_lo, frex_hi, 'CF_V1aV5p_G2-Sh')
        
        
    # Cross-Correlation Analysis
    if mode == 'tf_wave':
        
        # Looping over subjects to get data
#         for no, (sub1, sub2) in enumerate(zip(G1, G2)):
#             SourcesReco_Bursts(sub1, type_tr1, name1, no+1, mode)
#             SourcesReco_Bursts(sub2, type_tr2, name2, no+1, mode) 
            
        # Reading subjects' data in Src.Space
        for a in range(len(G1)):
            G1_pwr[a,:,:] = np.load('../../../Documents/3rd_Results/Subjects/TW/Pwr_%s_%s.npy' %(name1, a+1))
            G2_pwr[a,:,:] = np.load('../../../Documents/3rd_Results/Subjects/TW/Pwr_%s_%s.npy' %(name2, a+1))
            G1_wave[a,:,:,:] = np.load('../../../Documents/3rd_Results/Subjects/TW/Wave_%s_%s.npy' %(name1, a+1))
            G2_wave[a,:,:,:] = np.load('../../../Documents/3rd_Results/Subjects/TW/Wave_%s_%s.npy' %(name2, a+1))
            G1_twspeed[a,:,:] = np.load('../../../Documents/3rd_Results/Subjects/TW/TWSpeed_%s_%s.npy' %(name1, a+1))
            G2_twspeed[a,:,:] = np.load('../../../Documents/3rd_Results/Subjects/TW/TWSpeed_%s_%s.npy' %(name2, a+1))
              
        # Magnitude & Phase Decomposition
        G1_V5_time = np.mean(abs(np.mean(G1_wave[:,1,3:6,:],1)),0)             # Magnitude over time-points Alpha
        G1_V1_time = np.mean(abs(np.mean(G1_wave[:,0,3:6,:],1)),0)             # Magnitude over time-points Alpha
        G1_V5_ang = np.angle(np.mean(np.mean(G1_wave[:,1,3:6,:],1),0))         # Angle over time-points Alpha
        G1_V1_ang = np.angle(np.mean(np.mean(G1_wave[:,0,3:6,:],1),0))         # Angle over time-points Alpha
        G2_V5_time = np.mean(abs(np.mean(G2_wave[:,1,3:6,:],1)),0)             # Magnitude over time-points Alpha
        G2_V1_time = np.mean(abs(np.mean(G2_wave[:,0,3:6,:],1)),0)             # Magnitude over time-points Alpha
        G2_V5_ang = np.angle(np.mean(np.mean(G2_wave[:,1,3:6,:],1),0))         # Angle over time-points Alpha
        G2_V1_ang = np.angle(np.mean(np.mean(G2_wave[:,0,3:6,:],1),0))         # Angle over time-points Alpha
        
        # InterTrial Phase Consistency
        G1_frex = np.mean(abs(G1_wave[:,:,:,:]),(1,0))                          # Ch x frex x time-points
        G2_frex = np.mean(abs(G2_wave[:,:,:,:]),(1,0))                          # Ch x frex x time-points
#         G1_V5_frex = np.mean(abs(G1_wave[:,1,:,:]),0)                          # Ch x frex x time-points
#         G1_V1_frex = np.mean(abs(G1_wave[:,0,:,:]),0)                          # Ch x frex x time-points
#         G2_V5_frex = np.mean(abs(G2_wave[:,1,:,:]),0)                          # Ch x frex x time-points
#         G2_V1_frex = np.mean(abs(G2_wave[:,0,:,:]),0)                          # Ch x frex x time-points
        
        # Phase Locking Value & Speed Traveling Waves
        G1_tw_speed = np.mean(G1_twspeed,0)                                    # sub x frex x time-points
        G2_tw_speed = np.mean(G2_twspeed,0)                                    # sub x frex x time-points
           
                
        fig1 = plt.figure(figsize=(18,15))
        ax1 = fig1.add_subplot(2,2,1)
        stdev1 = np.std(G1_V5_time)
        ax1.plot(timex[350:525], G1_V5_time[350:525], label='V5')
        ax1.fill_between(timex[350:525], G1_V5_time[350:525]+stdev1, G1_V5_time[350:525]-stdev1,alpha=.1)
        stdev2 = np.std(G1_V1_time)
        ax1.plot(timex[350:525], G1_V1_time[350:525], label='V1')
        ax1.fill_between(timex[350:525], G1_V1_time[350:525]+stdev2, G1_V1_time[350:525]-stdev2,alpha=.1)
        ax1.set_ylim(0,0.25)
        ax1.tick_params(labelsize=20)
        ax1.set_ylabel('Magnitude', fontsize=20)
        # ax1.set_xlabel('Time(s)', fontsize=20)
        ax1.axvline(0, color='r') # Show Stim Onset
        ax1.set_title('VERUM - Alpha', fontsize=20)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, borderaxespad=0.)
        
        ax2 = fig1.add_subplot(2,2,2)
        stdev3 = np.std(G2_V5_time)
        ax2.plot(timex[350:525], G2_V5_time[350:525], label='V5')
        ax2.fill_between(timex[350:525], G2_V5_time[350:525]+stdev3, G2_V5_time[350:525]-stdev3,alpha=.1)
        stdev4 = np.std(G2_V1_time)
        ax2.plot(timex[350:525], G2_V1_time[350:525], label='V1')
        ax2.fill_between(timex[350:525], G2_V1_time[350:525]+stdev4, G2_V1_time[350:525]-stdev4,alpha=.1)
        ax2.set_ylim(0,0.25)
        ax2.tick_params(labelsize=20)
        ax2.set_ylabel('Magnitude', fontsize=20)
        # ax1.set_xlabel('Time(s)', fontsize=20)
        ax2.axvline(0, color='r') # Show Stim Onset
        ax2.set_title('SHAM - Alpha', fontsize=20)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, borderaxespad=0.)

        ax3 = fig1.add_subplot(2,2,3)
        ax3.plot(timex[350:525], G1_V5_ang[350:525], label='V5')
        ax3.plot(timex[350:525], G1_V1_ang[350:525], label='V1')
        # ax2.set_ylim(0,0.2)
        ax3.tick_params(labelsize=20)
        ax3.set_ylabel('Phase', fontsize=20)
        ax3.set_xlabel('Time(s)', fontsize=20)
        ax3.axvline(0, color='r') # Show Stim Onset
        ax3.set_title('VERUM - Alpha', fontsize=20)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, borderaxespad=0.)
        
        ax4 = fig1.add_subplot(2,2,4)
        ax4.plot(timex[350:525], G2_V5_ang[350:525], label='V5')
        ax4.plot(timex[350:525], G2_V1_ang[350:525], label='V1')
        # ax2.set_ylim(0,0.2)
        ax4.tick_params(labelsize=20)
        ax4.set_ylabel('Phase', fontsize=20)
        ax4.set_xlabel('Time(s)', fontsize=20)
        ax4.axvline(0, color='r') # Show Stim Onset
        ax4.set_title('SHAM - Alpha', fontsize=20)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, ncol=2, borderaxespad=0.)
        
        plt.show()
        fig1.savefig('Pwr_Phase_%s_%s' %(name1, name2))
        
        
        fig2 = plt.figure(figsize=(15,10))
        levels = MaxNLocator(nbins=50).tick_values(0.0, 0.3)
        levels1 = MaxNLocator(nbins=50).tick_values(-0.1, 0.1)
        
        ax5 = fig2.add_subplot(3,1,1)
        CS1 = ax5.contourf(timex[365:525], frex[:22], G1_frex[:22,365:525], cmap='RdBu_r', levels=levels, extend='both')
        fig2.colorbar(CS1, ax=ax5)
        ax5.tick_params(labelsize=20)
        ax5.set_ylabel('Freq.(Hz)', fontsize=20)
#         ax5.set_xlabel('Time(s)', fontsize=20)
        ax5.axvline(0, color='r') # Show Stim Onset
        ax5.set_title('VERUM', fontsize=20)

        ax6 = fig2.add_subplot(3,1,2)
        CS2 = ax6.contourf(timex[365:525], frex[:22], G2_frex[:22,365:525], cmap='RdBu_r', levels=levels, extend='both')
        fig2.colorbar(CS2, ax=ax6)
        ax6.tick_params(labelsize=20)
        ax6.set_ylabel('Freq.(Hz)', fontsize=20)
#         ax6.set_xlabel('Time(s)', fontsize=20)
        ax6.axvline(0, color='r') # Show Stim Onset
        ax6.set_title('SHAM', fontsize=20)
        
        ax7 = fig2.add_subplot(3,1,3)
        CS3 = ax7.contourf(timex[365:525], frex[:22], G2_frex[:22,365:525] - G1_frex[:22,365:525], cmap='RdBu_r', levels=levels1, extend='both')
        fig2.colorbar(CS3, ax=ax7)
        ax7.tick_params(labelsize=20)
        ax7.set_ylabel('Freq.(Hz)', fontsize=20)
        ax7.set_xlabel('Time(s)', fontsize=20)
        ax7.axvline(0, color='r') # Show Stim Onset
        ax7.set_title('SHAM-VERUM', fontsize=20)
        
#         ax8 = fig2.add_subplot(2,2,4)
#         CS4 = ax8.contourf(timex[350:525], frex[:22], G2_V5_frex[:22,365:525], cmap='RdBu_r', levels=levels, extend='both')
#         fig2.colorbar(CS4, ax=ax8)
#         ax8.tick_params(labelsize=20)
#         ax8.set_ylabel('Freq.(Hz)', fontsize=20)
#         ax8.set_xlabel('Time(s)', fontsize=20)
#         ax8.axvline(0, color='r') # Show Stim Onset
#         ax8.set_title('SHAM V5', fontsize=20)
        
        plt.tight_layout()
        plt.show()
        fig2.savefig('ITPC_%s_%s' %(name1, name2))
              
        
        fig3 = plt.figure(figsize=(15,8))
        levels_tws = MaxNLocator(nbins=50).tick_values(-0.2, 0.2)
        
        ax10 = fig3.add_subplot(1,2,1)
        CS5 = plt.contourf(timex[350:525], frex[:22], G1_tw_speed[:22,350:525], cmap='RdBu_r', levels=levels_tws, extend='both')
        fig3.colorbar(CS5, ax=ax10)
        ax10.set_title('Wave Speed m/s', fontsize=20)
        ax10.tick_params(labelsize=20)
        ax10.set_ylabel('Freq.(Hz)', fontsize=20)
        ax10.set_xlabel('Time(s)', fontsize=20)
        ax10.axvline(0, color='r') # Show Stim Onset
        
        ax11 = fig3.add_subplot(1,2,2)
        CS6 = plt.contourf(timex[350:525], frex[:22], G2_tw_speed[:22,350:525], cmap='RdBu_r', levels=levels_tws, extend='both')
        fig3.colorbar(CS6, ax=ax11)
        ax11.set_title('Wave Speed m/s', fontsize=20)
        ax11.tick_params(labelsize=20)
        ax11.set_ylabel('Freq.(Hz)', fontsize=20)
        ax11.set_xlabel('Time(s)', fontsize=20)
        ax11.axvline(0, color='r') # Show Stim Onset
        
        plt.show()
        fig3.savefig('TWS_%s_%s' %(name1, name2))

        
        fig4 = plt.figure(figsize=(15,10))
        levels_tws = MaxNLocator(nbins=50).tick_values(-0.3, 0.3)
        
        ax12 = fig4.add_subplot(3,1,1)
        CS7 = plt.contourf(timex[365:525], frex[:22], np.mean(G1_pwr[:,:22,365:525],0), cmap='RdBu_r', levels=levels_tws, extend='both')
        fig4.colorbar(CS7, ax=ax12)
        ax12.set_title('VERUM', fontsize=20)
        ax12.tick_params(labelsize=20)
        ax12.set_ylabel('Freq.(Hz)', fontsize=20)
#         ax12.set_xlabel('Time(s)', fontsize=20)
        ax12.axvline(0, color='r') # Show Stim Onset
        
        ax13 = fig4.add_subplot(3,1,2)
        CS8 = plt.contourf(timex[365:525], frex[:22], np.mean(G2_pwr[:,:22,365:525],0), cmap='RdBu_r', levels=levels_tws, extend='both')
        fig4.colorbar(CS8, ax=ax13)
        ax13.set_title('SHAM', fontsize=20)
        ax13.tick_params(labelsize=20)
        ax13.set_ylabel('Freq.(Hz)', fontsize=20)
#         ax13.set_xlabel('Time(s)', fontsize=20)
        ax13.axvline(0, color='r') # Show Stim Onset
        
        ax14 = fig4.add_subplot(3,1,3)
        CS9 = plt.contourf(timex[365:525], frex[:22], np.mean(G2_pwr[:,:22,365:525],0) - np.mean(G1_pwr[:,:22,365:525],0),
                           cmap='RdBu_r', levels=levels_tws, extend='both')
        fig4.colorbar(CS9, ax=ax14)
        ax14.set_title('SHAM-VERUM', fontsize=20)
        ax14.tick_params(labelsize=20)
        ax14.set_ylabel('Freq.(Hz)', fontsize=20)
        ax14.set_xlabel('Time(s)', fontsize=20)
        ax14.axvline(0, color='r') # Show Stim Onset
        
        plt.tight_layout()
        plt.show()
        fig4.savefig('TF_%s_%s' %(name1, name2))

        