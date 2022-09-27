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

from SourcesReco import *
from CrossCor import *

def Group_Src (G1, G2, type_tr1, type_tr2, name1, name2, mode):
# G1: list of paths corresponding to 1st group
# G2: list of paths corresponding to 2nd group
# type_tr: Correct='stim_cor' or Incorrect='stim_inc'
# name1: name to save the output image
# name2: name to save the output image
# mode: Either band_power, tf_power, phase or cross_freq analysis of the electrodes

    # Initialize Group TimeFreq /Phase Arrays

    G1_theta_stc = list()
    G1_alpha_stc = list()
    G1_betha_stc = list()
    G1_gamma_stc = list()
    G2_theta_stc = list()
    G2_alpha_stc = list()
    G2_betha_stc = list()
    G2_gamma_stc = list()
    G1_V1_alpha_stc = list()
    G1_V5_alpha_stc = list()
    G1_V1_gamma_stc = list()
    G1_V5_gamma_stc = list()
    G2_V1_alpha_stc = list()
    G2_V5_alpha_stc = list()
    G2_V1_gamma_stc = list()
    G2_V5_gamma_stc = list()
    G1_theta_np = np.empty([])
    G1_alpha_np = np.empty([])
    G1_betha_np = np.empty([])
    G1_gamma_np = np.empty([])
    G2_theta_np = np.empty([])
    G2_alpha_np = np.empty([])
    G2_betha_np = np.empty([])
    G2_gamma_np = np.empty([])
    G1_con = np.empty([2,2,40,750,15])
    G2_con = np.empty([2,2,40,750,15])
    G1_cau = np.empty([2,2,4,750,15])
    G2_cau = np.empty([2,2,4,750,15])
    G1_tf_pwr = np.empty([15,40,750])
    G2_tf_pwr = np.empty([15,40,750])
    G1_cf = np.empty([5,6,2,15])
    G2_cf = np.empty([5,6,2,15])
    G1_cc = np.empty([250,15])
    G2_cc = np.empty([250,15])
    G1_data_alpha = np.empty([2,750,15], dtype=complex)
    G2_data_alpha = np.empty([2,750,15], dtype=complex)
    G1_data_gamma = np.empty([2,750,15], dtype=complex)
    G2_data_gamma = np.empty([2,750,15], dtype=complex)
    
    # Times of interest
    min_time = -1.5
    max_time = 1.5
    num_time = 750
    timex = np.linspace(min_time, max_time, num_time)
    
    # Frequency Parameters
    min_freq = 2
    max_freq = 42
    num_freq = 40
    frex = np.linspace(min_freq, max_freq, num_freq)  

        # Plotting Band Power across Time
    if mode == 'psd_v1v5':
        
#         # Looping over subjects to get data --> To comment when data has already been computed
#         for no, (sub1, sub2) in enumerate(zip(G1, G2)):
#             SourcesReco(sub1, type_tr1, name1, no+1, mode)
#             SourcesReco(sub2, type_tr2, name2, no+1, mode) 
        
        from IPython.display import Image
        from mne.datasets import spm_face
        data_path = spm_face.data_path()   
        subjects_dir = data_path + '/subjects'
        
        for a in range(len(G1)): # No need of morphing 'cause it is always the same mesh
            
            G1_V1_alpha_sub = mne.read_source_estimate('Pwr_V1_0_%s_%s-stc.h5' %(name1, a+1)) # Alpha V1
            G1_V1_alpha_stc = np.append(G1_V1_alpha_sub, G1_V1_alpha_stc)
            G1_V5_alpha_sub = mne.read_source_estimate('Pwr_V5_0_%s_%s-stc.h5' %(name1, a+1)) # Alpha V5
            G1_V5_alpha_stc = np.append(G1_V5_alpha_sub, G1_V5_alpha_stc)
            G1_V1_gamma_sub = mne.read_source_estimate('Pwr_V1_1_%s_%s-stc.h5' %(name1, a+1)) # Gamma V1
            G1_V1_gamma_stc = np.append(G1_V1_gamma_sub, G1_V1_gamma_stc)
            G1_V5_gamma_sub = mne.read_source_estimate('Pwr_V5_1_%s_%s-stc.h5' %(name1, a+1)) # Gamma V5
            G1_V5_gamma_stc = np.append(G1_V5_gamma_sub, G1_V5_gamma_stc)
            
            G2_V1_alpha_sub = mne.read_source_estimate('Pwr_V1_0_%s_%s-stc.h5' %(name2, a+1)) # Alpha V1 Post
            G2_V1_alpha_stc = np.append(G2_V1_alpha_sub, G2_V1_alpha_stc)
            G2_V5_alpha_sub = mne.read_source_estimate('Pwr_V5_0_%s_%s-stc.h5' %(name2, a+1)) # Alpha V5 Post
            G2_V5_alpha_stc = np.append(G2_V5_alpha_sub, G2_V5_alpha_stc)
            G2_V1_gamma_sub = mne.read_source_estimate('Pwr_V1_1_%s_%s-stc.h5' %(name2, a+1)) # Gamma V1 Post
            G2_V1_gamma_stc = np.append(G2_V1_gamma_sub, G2_V1_gamma_stc)
            G2_V5_gamma_sub = mne.read_source_estimate('Pwr_V5_1_%s_%s-stc.h5' %(name2, a+1)) # Gamma V5 Post
            G2_V5_gamma_stc = np.append(G2_V5_gamma_sub, G2_V5_gamma_stc)
                       
        # Averaging data through subjects
        G1_V1_alpha_data = np.mean([s.data for s in G1_V1_alpha_stc], axis=0)
        G1_V5_alpha_data = np.mean([s.data for s in G1_V5_alpha_stc], axis=0)
        G1_V1_gamma_data = np.mean([s.data for s in G1_V1_gamma_stc], axis=0)
        G1_V5_gamma_data = np.mean([s.data for s in G1_V5_gamma_stc], axis=0)
                                             
        G2_V1_alpha_data = np.mean([s.data for s in G2_V1_alpha_stc], axis=0)
        G2_V5_alpha_data = np.mean([s.data for s in G2_V5_alpha_stc], axis=0)
        G2_V1_gamma_data = np.mean([s.data for s in G2_V1_gamma_stc], axis=0)
        G2_V5_gamma_data = np.mean([s.data for s in G2_V5_gamma_stc], axis=0)
        
        # Computing Source estimate of the average --> Every Source recons. is different due to different timings = Frex Bins
        G1_V1_alpha_avg = mne.SourceEstimate(G1_V1_alpha_data, G1_V1_alpha_stc[0].vertices, G1_V1_alpha_stc[0].tmin,
                                          G1_V1_alpha_stc[0].tstep, subject='spm')
        G1_V5_alpha_avg = mne.SourceEstimate(G1_V5_alpha_data, G1_V5_alpha_stc[0].vertices, G1_V5_alpha_stc[0].tmin,
                                          G1_V5_alpha_stc[0].tstep, subject='spm')
        G1_V1_gamma_avg = mne.SourceEstimate(G1_V1_gamma_data, G1_V1_gamma_stc[0].vertices, G1_V1_gamma_stc[0].tmin,
                                          G1_V1_gamma_stc[0].tstep, subject='spm')
        G1_V5_gamma_avg = mne.SourceEstimate(G1_V5_gamma_data, G1_V5_gamma_stc[0].vertices, G1_V5_gamma_stc[0].tmin,
                                          G1_V5_gamma_stc[0].tstep, subject='spm')
        
        G2_V1_alpha_avg = mne.SourceEstimate(G2_V1_alpha_data, G2_V1_alpha_stc[0].vertices, G2_V1_alpha_stc[0].tmin,
                                          G2_V1_alpha_stc[0].tstep, subject='spm')
        G2_V5_alpha_avg = mne.SourceEstimate(G2_V5_alpha_data, G2_V5_alpha_stc[0].vertices, G2_V5_alpha_stc[0].tmin,
                                          G2_V5_alpha_stc[0].tstep, subject='spm')
        G2_V1_gamma_avg = mne.SourceEstimate(G2_V1_gamma_data, G2_V1_gamma_stc[0].vertices, G2_V1_gamma_stc[0].tmin,
                                          G2_V1_gamma_stc[0].tstep, subject='spm')
        G2_V5_gamma_avg = mne.SourceEstimate(G2_V5_gamma_data, G2_V5_gamma_stc[0].vertices, G2_V5_gamma_stc[0].tmin,
                                          G2_V5_gamma_stc[0].tstep, subject='spm')
        
        # Computing Post-Pre Diff
        Sub_V1_alpha = mne.SourceEstimate(G2_V1_alpha_data-G1_V1_alpha_data, G1_V1_alpha_stc[0].vertices, G1_V1_alpha_stc[0].tmin,
                                          G1_V1_alpha_stc[0].tstep, subject='spm')
        Sub_V5_alpha = mne.SourceEstimate(G2_V5_alpha_data-G1_V5_alpha_data, G1_V5_alpha_stc[0].vertices, G1_V5_alpha_stc[0].tmin,
                                          G1_V5_alpha_stc[0].tstep, subject='spm')
        Sub_V1_gamma = mne.SourceEstimate(G2_V1_gamma_data-G1_V1_gamma_data, G1_V1_gamma_stc[0].vertices, G1_V1_gamma_stc[0].tmin,
                                          G1_V1_gamma_stc[0].tstep, subject='spm')
        Sub_V5_gamma = mne.SourceEstimate(G2_V5_gamma_data-G1_V5_gamma_data, G1_V5_gamma_stc[0].vertices, G1_V5_gamma_stc[0].tmin,
                                          G1_V5_gamma_stc[0].tstep, subject='spm')       
        
        # Stack V1-V5 activation vertices in a single list to make a single figure of the activation
        G1_V1V5_a_vex = G1_V1_alpha_stc[0].vertices
        G1_V1V5_a_vex[1] = np.sort(np.concatenate((G1_V1_alpha_stc[0].vertices[1],G1_V5_alpha_stc[0].vertices[1]), axis=0))
        G1_V1V5_g_vex = G1_V1_alpha_stc[0].vertices
        G1_V1V5_g_vex[1] = np.sort(np.concatenate((G1_V1_gamma_stc[0].vertices[1],G1_V5_gamma_stc[0].vertices[1]), axis=0))
        G2_V1V5_a_vex = G2_V1_alpha_stc[0].vertices
        G2_V1V5_a_vex[1] = np.sort(np.concatenate((G2_V1_alpha_stc[0].vertices[1],G2_V5_alpha_stc[0].vertices[1]), axis=0))
        G2_V1V5_g_vex = G2_V1_alpha_stc[0].vertices
        G2_V1V5_g_vex[1] = np.sort(np.concatenate((G2_V1_gamma_stc[0].vertices[1],G2_V5_gamma_stc[0].vertices[1]), axis=0))
        
        G1_V1V5_alpha = mne.SourceEstimate(np.sort(np.concatenate((G1_V1_alpha_data,G1_V5_alpha_data))), G1_V1V5_a_vex,
                                           G1_V1_alpha_stc[0].tmin, G1_V1_alpha_stc[0].tstep, subject='spm')
        G1_V1V5_gamma = mne.SourceEstimate(np.sort(np.concatenate((G1_V1_gamma_data,G1_V5_gamma_data))), G1_V1V5_g_vex, 
                                           G1_V5_gamma_stc[0].tmin, G1_V5_gamma_stc[0].tstep, subject='spm')
        G2_V1V5_alpha = mne.SourceEstimate(np.sort(np.concatenate((G2_V1_alpha_data,G2_V5_alpha_data))), G2_V1V5_a_vex,
                                           G2_V1_alpha_stc[0].tmin, G2_V1_alpha_stc[0].tstep, subject='spm')
        G2_V1V5_gamma = mne.SourceEstimate(np.sort(np.concatenate((G2_V1_gamma_data,G2_V5_gamma_data))), G2_V1V5_g_vex, 
                                           G2_V5_gamma_stc[0].tmin, G2_V5_gamma_stc[0].tstep, subject='spm')
        
        # 3D Plotting 
#         G1_V1_brain_a = G1_V1_alpha_avg.plot(views='cau', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w')#, clim=dict(kind='percent', lims=[7, 10, 13]))
#         G1_V1_brain_a.save_image('Pwr_V1_0_%s.jpg' %name1)
#         G1_V1_brain_g = G1_V1_gamma_avg.plot(views='cau', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w')#clim={'kind':'value','lims': [30.,36.,42.]})
#         G1_V1_brain_g.save_image('Pwr_V1_1_%s.jpg' %name1)
        
#         G2_V1_brain_a = G2_V1_alpha_avg.plot(views='cau', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w')
#         G2_V1_brain_a.save_image('Pwr_V1_0_%s.jpg' %name2)        
#         G2_V1_brain_g = G2_V1_gamma_avg.plot(views='cau', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w')
#         G2_V1_brain_g.save_image('Pwr_V1_1_%s.jpg' %name2)
        
#         Sub_V1_brain_a = Sub_V1_alpha.plot(views='cau', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w')
#         Sub_V1_brain_a.save_image('Pwr_V1_0_%s_Bsl.jpg' %name2)         
#         Sub_V1_brain_g = Sub_V1_gamma.plot(views='cau', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w')
#         Sub_V1_brain_g.save_image('Pwr_V1_1_%s_Bsl.jpg' %name2)
                                        
#         G1_V5_brain_a = G1_V5_alpha_avg.plot(views='par', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w')
#         G1_V5_brain_a.save_image('Pwr_V5_0_%s.jpg' %name1)
#         G1_V5_brain_g = G1_V5_gamma_avg.plot(views='par', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w')
#         G1_V5_brain_g.save_image('Pwr_V5_1_%s.jpg' %name1)
        
#         G2_V5_brain_a = G2_V5_alpha_avg.plot(views='par', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w')
#         G2_V5_brain_a.save_image('Pwr_V5_0_%s.jpg' %name2)        
#         G2_V5_brain_g = G2_V5_gamma_avg.plot(views='par', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w')
#         G2_V5_brain_g.save_image('Pwr_V5_1_%s.jpg' %name2)
   
#         Sub_V5_brain_a = Sub_V5_alpha.plot(views='par', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w')
#         Sub_V5_brain_a.save_image('Pwr_V5_0_%s_Bsl.jpg' %name2)         
#         Sub_V5_brain_g = Sub_V5_gamma.plot(views='par', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w')
#         Sub_V5_brain_g.save_image('Pwr_V5_1_%s_Bsl.jpg' %name2)
           
#         G1_V1V5_a = G1_V1V5_alpha.plot(views='cau', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='rainbow', background='w',
#                                        clim=dict(kind='value', lims=[1e-26, 5e-26, 10e-26]))            
#         G1_V1V5_a.save_image('Pwr_V1V5_0_%s.jpg' %name1)         
#         G1_V1V5_g = G1_V1V5_gamma.plot(views='cau', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w',
#                                        clim=dict(kind='value', lims=[0, 5e-26, 10e-26]))
#         G1_V1V5_g.save_image('Pwr_V1V5_1_%s.jpg' %name1) 
#         G2_V1V5_a = G2_V1V5_alpha.plot(views='cau', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w',
#                                        clim=dict(kind='value', lims=[0, 5e-26, 10e-26]))
#         G2_V1V5_a.save_image('Pwr_V1V5_0_%s.jpg' %name2)         
#         G2_V1V5_g = G2_V1V5_gamma.plot(views='cau', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis', background='w',
#                                        clim=dict(kind='value', lims=[0, 5e-26, 10e-26]))
#         G2_V1V5_g.save_image('Pwr_V1V5_1_%s.jpg' %name2)

        ## Publishable Figures
        # Normalize in case it is needed intensity of PSD = Single value for Freq. Band
        from scipy import stats
        G1_V1V5_alpha.data[:,0] = stats.zscore(np.mean(G1_V1V5_alpha.data, axis=1)) # Add average of all Fr to the first position of the SourceEstimate Data
        G1_V1V5_gamma.data[:,0] = stats.zscore(np.mean(G1_V1V5_gamma.data, axis=1)) # Each timepoint of the SourceEstimate is actually one frequency bin
        
        G1_V1V5_a_cau = G1_V1V5_alpha.plot(initial_time=0, views='cau', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='rainbow', background='w',
                                      colorbar=False, time_viewer=False, show_traces=False)       
        G1_V1V5_a_par = G1_V1V5_alpha.plot(initial_time=0, views='par', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='rainbow', background='w',
                                      colorbar=False, time_viewer=False, show_traces=False)
        G1_V1V5_g_cau = G1_V1V5_gamma.plot(initial_time=0, views='cau', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='rainbow', background='w',
                                      colorbar=False, time_viewer=False, show_traces=False)       
        G1_V1V5_g_par = G1_V1V5_gamma.plot(initial_time=0, views='par', hemi='rh', size=(1080, 920), subjects_dir=subjects_dir, colormap='rainbow', background='w',
                                      colorbar=False, time_viewer=False, show_traces=False)
        
        def pubfig(d1, d2, name):
            cau = d1.screenshot()
            par = d2.screenshot()

            def onlyimg(screenshot):
                nonwhite_pix = (screenshot != 255).any(-1)
                nonwhite_row = nonwhite_pix.any(1)
                nonwhite_col = nonwhite_pix.any(0)
                cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
                return cropped_screenshot

            crop_cau = onlyimg(cau)
            crop_par = onlyimg(par) 

            fig = plt.figure(figsize=(15.0, 13.0))
            ax1 = fig.add_subplot(1,2,1)
            ax1.imshow(crop_par)
            ax1.axis('off')
            ax2 = fig.add_subplot(1,2,2)
            ax2.imshow(crop_cau)
            ax2.axis('off')
            # add a vertical colorbar with the same properties as the 3D one
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes('right', size='5%', pad=0.2)
            clim = dict(kind='value', lims=[0,3,4])
            cbar = mne.viz.plot_brain_colorbar(cax, clim, 'rainbow', label='Frequency Activation (Hz)')

            fig.savefig('Bsl_Src_%s.jpg' %name)
            
        pubfig(G1_V1V5_a_cau, G1_V1V5_a_par, 'alpha')
        pubfig(G1_V1V5_g_cau, G1_V1V5_g_par, 'gamma')
    
    # Plotting Band Power across Time
    if mode == 'psd':
        
        # Looping over subjects to get data --> To comment when data has already been computed
        for no, (sub1, sub2) in enumerate(zip(G1, G2)):
            SourcesReco(sub1, type_tr1, name1, no+1, mode)
            SourcesReco(sub2, type_tr2, name2, no+1, mode) 
        
        from IPython.display import Image
        from mne.datasets import spm_face
        data_path = spm_face.data_path()   
        subjects_dir = data_path + '/subjects'
        
        for a in range(len(G1)): # No need of morphing 'cause it is always the same mesh
            
            G1_theta_sub = mne.read_source_estimate('Pwr_0_%s_%s-stc.h5' %(name1, a+1)) # Theta
            G1_theta_stc = np.append(G1_theta_sub, G1_theta_stc)
            G1_alpha_sub = mne.read_source_estimate('Pwr_1_%s_%s-stc.h5' %(name1, a+1)) # Alpha
            G1_alpha_stc = np.append(G1_alpha_sub, G1_alpha_stc)
            G1_betha_sub = mne.read_source_estimate('Pwr_2_%s_%s-stc.h5' %(name1, a+1)) # Betha
            G1_betha_stc = np.append(G1_betha_sub, G1_betha_stc)
            G1_gamma_sub = mne.read_source_estimate('Pwr_3_%s_%s-stc.h5' %(name1, a+1)) # Gamma
            G1_gamma_stc = np.append(G1_gamma_sub, G1_gamma_stc)
            
            G2_theta_sub = mne.read_source_estimate('Pwr_0_%s_%s-stc.h5' %(name2, a+1))
            G2_theta_stc = np.append(G2_theta_sub, G2_theta_stc)
            G2_alpha_sub = mne.read_source_estimate('Pwr_1_%s_%s-stc.h5' %(name2, a+1))
            G2_alpha_stc = np.append(G2_alpha_sub, G2_alpha_stc)
            G2_betha_sub = mne.read_source_estimate('Pwr_2_%s_%s-stc.h5' %(name2, a+1))
            G2_betha_stc = np.append(G2_betha_sub, G2_betha_stc)
            G2_gamma_sub = mne.read_source_estimate('Pwr_3_%s_%s-stc.h5' %(name2, a+1))
            G2_gamma_stc = np.append(G2_gamma_sub, G2_gamma_stc)
                       
        # Averaging data through subjects
        G1_theta_data = np.mean([s.data for s in G1_theta_stc], axis=0)
        G1_alpha_data = np.mean([s.data for s in G1_alpha_stc], axis=0)
        G1_betha_data = np.mean([s.data for s in G1_betha_stc], axis=0)
        G1_gamma_data = np.mean([s.data for s in G1_gamma_stc], axis=0)
                                     
        G2_theta_data = np.mean([s.data for s in G2_theta_stc], axis=0)
        G2_alpha_data = np.mean([s.data for s in G2_alpha_stc], axis=0)
        G2_betha_data = np.mean([s.data for s in G2_betha_stc], axis=0)
        G2_gamma_data = np.mean([s.data for s in G2_gamma_stc], axis=0)
        
        # Computing Source estimate of the average --> Every Source recons. is different due to different timings = Frex Bins
        G1_theta_avg = mne.SourceEstimate(G1_theta_data, G1_theta_stc[0].vertices, G1_theta_stc[0].tmin,
                                          G1_theta_stc[0].tstep, subject='spm')
        G1_alpha_avg = mne.SourceEstimate(G1_alpha_data, G1_alpha_stc[0].vertices, G1_alpha_stc[0].tmin,
                                          G1_alpha_stc[0].tstep, subject='spm')
        G1_betha_avg = mne.SourceEstimate(G1_betha_data, G1_betha_stc[0].vertices, G1_betha_stc[0].tmin,
                                          G1_betha_stc[0].tstep, subject='spm')
        G1_gamma_avg = mne.SourceEstimate(G1_gamma_data, G1_gamma_stc[0].vertices, G1_gamma_stc[0].tmin,
                                          G1_gamma_stc[0].tstep, subject='spm')
        
        G2_theta_avg = mne.SourceEstimate(G2_theta_data, G2_theta_stc[0].vertices, G2_theta_stc[0].tmin,
                                          G2_theta_stc[0].tstep, subject='spm')
        G2_alpha_avg = mne.SourceEstimate(G2_alpha_data, G2_alpha_stc[0].vertices, G2_alpha_stc[0].tmin,
                                          G2_alpha_stc[0].tstep, subject='spm')
        G2_betha_avg = mne.SourceEstimate(G2_betha_data, G2_betha_stc[0].vertices, G2_betha_stc[0].tmin,
                                          G2_betha_stc[0].tstep, subject='spm')
        G2_gamma_avg = mne.SourceEstimate(G2_gamma_data, G2_gamma_stc[0].vertices, G2_gamma_stc[0].tmin,
                                          G2_gamma_stc[0].tstep, subject='spm')
        
        # Computing Post-Pre Diff
        Sub_theta = mne.SourceEstimate(G2_theta_data-G1_theta_data, G1_theta_stc[0].vertices, G1_theta_stc[0].tmin,
                                          G1_theta_stc[0].tstep, subject='spm')
        Sub_alpha = mne.SourceEstimate(G2_alpha_data-G1_alpha_data, G1_alpha_stc[0].vertices, G1_alpha_stc[0].tmin,
                                          G1_alpha_stc[0].tstep, subject='spm')
        Sub_betha = mne.SourceEstimate(G2_betha_data-G1_betha_data, G1_betha_stc[0].vertices, G1_betha_stc[0].tmin,
                                          G1_betha_stc[0].tstep, subject='spm')
        Sub_gamma = mne.SourceEstimate(G2_gamma_data-G1_gamma_data, G1_gamma_stc[0].vertices, G1_gamma_stc[0].tmin,
                                          G1_gamma_stc[0].tstep, subject='spm')
        
        
        # 3D Plotting 
        G1_brain_t = G1_theta_avg.plot(views='lat', hemi='split', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis')#, clim={'kind':'value','lims': [2., 4., 6.]})
        #G1_brain_t = brain_t.scale_data_colormap(fmin=2, fmid=4, fmax=6, transparent=True)
        G1_brain_t.save_image('Pwr_0_%s.jpg' %name1)
        #mlab.show()
        G1_brain_a = G1_alpha_avg.plot(views='lat', hemi='split', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis')#, clim=dict(kind='percent', lims=[7, 10, 13]))
        G1_brain_a.save_image('Pwr_1_%s.jpg' %name1)
        G1_brain_b = G1_betha_avg.plot(views='lat', hemi='split', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis')#clim={'kind':'value','lims': [14.,21.,29.]})
        G1_brain_b.save_image('Pwr_2_%s.jpg' %name1)     
        G1_brain_g = G1_gamma_avg.plot(views='lat', hemi='split', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis')#clim={'kind':'value','lims': [30.,36.,42.]})
        G1_brain_g.save_image('Pwr_3_%s.jpg' %name1)
        
        G2_brain_t = G2_theta_avg.plot(views='lat', hemi='split', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis')
        G2_brain_t.save_image('Pwr_0_%s.jpg' %name2)
        G2_brain_a = G2_alpha_avg.plot(views='lat', hemi='split', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis')
        G2_brain_a.save_image('Pwr_1_%s.jpg' %name2)      
        G2_brain_b = G2_betha_avg.plot(views='lat', hemi='split', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis')
        G2_brain_b.save_image('Pwr_2_%s.jpg' %name2)     
        G2_brain_g = G2_gamma_avg.plot(views='lat', hemi='split', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis')
        G2_brain_g.save_image('Pwr_3_%s.jpg' %name2)
        
        Sub_brain_t = Sub_theta.plot(views='lat', hemi='split', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis')
        Sub_brain_t.save_image('Pwr_0_%s_Bsl.jpg' %name2)
        Sub_brain_a = Sub_alpha.plot(views='lat', hemi='split', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis')
        Sub_brain_a.save_image('Pwr_1_%s_Bsl.jpg' %name2)     
        Sub_brain_b = Sub_betha.plot(views='lat', hemi='split', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis')
        Sub_brain_b.save_image('Pwr_2_%s_Bsl.jpg' %name2)     
        Sub_brain_g = Sub_gamma.plot(views='lat', hemi='split', size=(1080, 920), subjects_dir=subjects_dir, colormap='viridis')
        Sub_brain_g.save_image('Pwr_3_%s_Bsl.jpg' %name2)
    
    
    # Plotting Band Power across Time --> TO BE CHECKED: 
    ## Should one re compute a mne.SourceEstimate rather than copying the data structure of one of the subjects as basis for the group analysis?
    if mode == 'band_power':

#          # Looping over subjects to get data --> To comment when data has already been computed
#         for no, (sub1, sub2) in enumerate(zip(G1, G2)):
#             SourcesReco(sub1, type_tr1, name1, no+1, mode)
#             SourcesReco(sub2, type_tr2, name2, no+1, mode) 
        
        # Reading subject's data in Src.Space
        for a in range(len(G1)):
#           Real path to find the individual files
#             '../../../Documents/Sources_May19/Power/Data/G4/
            G1_theta_sub = mne.read_source_estimate('Pwr_0_%s_%s-stc.h5' %(name1, a+1)) # Theta
#             print(G1_theta_sub)
#             print(G1_theta_sub.data.shape)
#             print(type(G1_theta_sub))
            G1_theta_np = np.append(G1_theta_sub, G1_theta_np)
            G1_alpha_sub = mne.read_source_estimate('Pwr_1_%s_%s-stc.h5' %(name1, a+1)) # Alpha
            G1_alpha_np = np.append(G1_alpha_sub, G1_alpha_np)
            G1_betha_sub = mne.read_source_estimate('Pwr_2_%s_%s-stc.h5' %(name1, a+1)) # Betha
            G1_betha_np = np.append(G1_betha_sub, G1_betha_np)
            G1_gamma_sub = mne.read_source_estimate('Pwr_3_%s_%s-stc.h5' %(name1, a+1)) # Gamma
            G1_gamma_np = np.append(G1_gamma_sub, G1_gamma_np)
            
            G2_theta_sub = mne.read_source_estimate('Pwr_0_%s_%s-stc.h5' %(name2, a+1))
            G2_theta_np = np.append(G2_theta_sub, G2_theta_np)
            G2_alpha_sub = mne.read_source_estimate('Pwr_1_%s_%s-stc.h5' %(name2, a+1))
            G2_alpha_np = np.append(G2_alpha_sub, G2_alpha_np)
            G2_betha_sub = mne.read_source_estimate('Pwr_2_%s_%s-stc.h5' %(name2, a+1))
            G2_betha_np = np.append(G2_betha_sub, G2_betha_np)
            G2_gamma_sub = mne.read_source_estimate('Pwr_3_%s_%s-stc.h5' %(name2, a+1))
            G2_gamma_np = np.append(G2_gamma_sub, G2_gamma_np)
            
        # Removing parasite value in position 16th of the array given the preallocation of empty space
        G1_theta_np = G1_theta_np[:15]
        G1_alpha_np = G1_alpha_np[:15]
        G1_betha_np = G1_betha_np[:15]
        G1_gamma_np = G1_gamma_np[:15]
               
        # Copying data structure and Data Pwr matrix [sources x frex bins]
        G1_theta_avg = G1_theta_np[0]
        G1_theta_data = G1_theta_np[0].data
        G1_alpha_avg = G1_alpha_np[0]
        G1_alpha_data = G1_alpha_np[0].data
        G1_betha_avg = G1_betha_np[0]
        G1_betha_data = G1_betha_np[0].data
        G1_gamma_avg = G1_gamma_np[0]
        G1_gamma_data = G1_gamma_np[0].data
        
        G2_theta_avg = G2_theta_np[0]
        G2_theta_data = G2_theta_np[0].data
        G2_alpha_avg = G2_alpha_np[0]
        G2_alpha_data = G2_alpha_np[0].data
        G2_betha_avg = G2_betha_np[0]
        G2_betha_data = G2_betha_np[0].data
        G2_gamma_avg = G2_gamma_np[0]
        G2_gamma_data = G2_gamma_np[0].data
        
        print('uno', G1_alpha_data.shape)
        
        # Stacking all the Pwr values from ALL SUBJECTS in a single 3D matrix
        for b in range(len(G1)-1):
            G1_theta_data = np.dstack((G1_theta_data, G1_theta_np[b+1].data))
            G1_alpha_data = np.dstack((G1_alpha_data, G1_alpha_np[b+1].data))
            G1_betha_data = np.dstack((G1_betha_data, G1_betha_np[b+1].data))
            G1_gamma_data = np.dstack((G1_gamma_data, G1_gamma_np[b+1].data))
            
            G2_theta_data = np.dstack((G2_theta_data, G2_theta_np[b+1].data))
            G2_alpha_data = np.dstack((G2_alpha_data, G2_alpha_np[b+1].data))
            G2_betha_data = np.dstack((G2_betha_data, G2_betha_np[b+1].data))
            G2_gamma_data = np.dstack((G2_gamma_data, G2_gamma_np[b+1].data))
            
        print('dos', G1_alpha_data.shape)
        
        # Assigning Averaged Power over Subjects to data structure
        G1_theta_avg.data = np.mean(G1_theta_data, axis=2)
        G1_alpha_avg.data = np.mean(G1_alpha_data, axis=2)
        G1_betha_avg.data = np.mean(G1_betha_data, axis=2)
        G1_gamma_avg.data = np.mean(G1_gamma_data, axis=2)
                                     
        G2_theta_avg.data = np.mean(G2_theta_data, axis=2)
        G2_alpha_avg.data = np.mean(G2_alpha_data, axis=2)
        G2_betha_avg.data = np.mean(G2_betha_data, axis=2)
        G2_gamma_avg.data = np.mean(G2_gamma_data, axis=2)
        
        print('tres', G1_alpha_avg.data.shape)
        
        # Calculating G2-G1 (Pos-Pre) Difference
        Sub_theta = G1_theta_np[0]
        Sub_theta.data = G2_theta_avg.data - G1_theta_avg.data
        Sub_alpha = G1_alpha_np[0]
        Sub_alpha.data = G2_alpha_avg.data - G1_alpha_avg.data
        Sub_betha = G1_betha_np[0]
        Sub_betha.data = G2_betha_avg.data - G1_betha_avg.data
        Sub_gamma = G1_gamma_np[0]
        Sub_gamma.data = G2_gamma_avg.data - G1_gamma_avg.data
                    
        # Plotting
        fig1 = plt.figure(figsize=(15.0, 13.0))
        
        ax1 = fig1.add_subplot(2,1,1)
        ax1.plot(G1_theta_avg.times[350:500], G1_theta_avg.data.mean(axis=0)[350:500], label='Theta')
        ax1.plot(G1_alpha_avg.times[350:500], G1_alpha_avg.data.mean(axis=0)[350:500], label='Alpha')
        ax1.plot(G1_betha_avg.times[350:500], G1_betha_avg.data.mean(axis=0)[350:500], label='Betha')
        ax1.plot(G1_gamma_avg.times[350:500], G1_gamma_avg.data.mean(axis=0)[350:500], label='Gamma')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        ax1.set_xlabel('Time (ms)')
        ax1.set_ylabel('Power')
        ax1.set_title('Pre')
        
        ax2 = fig1.add_subplot(2,1,2)
        ax2.plot(G2_theta_avg.times[350:500], G2_theta_avg.data.mean(axis=0)[350:500], label='Theta')
        ax2.plot(G2_alpha_avg.times[350:500], G2_alpha_avg.data.mean(axis=0)[350:500], label='Alpha')
        ax2.plot(G2_betha_avg.times[350:500], G2_betha_avg.data.mean(axis=0)[350:500], label='Betha')
        ax2.plot(G2_gamma_avg.times[350:500], G2_gamma_avg.data.mean(axis=0)[350:500], label='Gamma')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        ax2.set_xlabel('Time (ms)')
        ax2.set_ylabel('Power')
        ax2.set_title('Pos')
              
        fig1.savefig('Src_Pwr_%s_%s' %(name1, name2))
        #plt.show()
        
        # 3D Plotting --> Videos of Power
#         from IPython.display import Image
#         from mne.datasets import spm_face
#         data_path = spm_face.data_path()   
#         subjects_dir = data_path + '/subjects'
#         brain = G2_alpha_avg.plot('spm', initial_time=0, time_viewer=1, subjects_dir=subjects_dir, hemi='split') # Videos of power 

# Plotting  TF Power (ROI sources)
    if mode == 'tf_hilb':
        
        from mne.datasets import spm_face
        data_path = spm_face.data_path()
        src = data_path + '/subjects/spm/bem/spm-oct-6-src.fif'
        bem = data_path + '/subjects/spm/bem/spm-5120-5120-5120-bem-sol.fif'
        
        def hilb (sub, fmax, fmin):
            cov = mne.compute_covariance(sub, tmin=-.300, tmax=-.100, method='empirical')
            fwd = mne.make_forward_solution(sub.info, 'fsaverage', src, bem, eeg=True)
            inv = make_inverse_operator(sub.info, fwd, cov, loose=0.2, depth=0.8)
            suf = sub.filter(fmax, fmin, fir_design='firwin')
            suh = suf.apply_hilbert()
            stcs = apply_inverse_epochs(suh, inv, lambda2=1/9, pick_ori='normal', method='MNE')
            names = ['rh.V1','rh.MT'] # Read labels of interest
            labels_parc = [mne.read_label(data_path + '/subjects/spm/label/%s.label' % name) for name in names]
            label_ts = mne.extract_label_time_course(stcs, labels_parc, inv['src'], mode='pca_flip', 
                                                             allow_empty=True, return_generator=False)
            data = np.array(label_ts, dtype=complex)
            print(data.shape)
            del sub, fmax, fmin
            return np.mean(data, axis=0)
    
        # Looping over subjects to get data --> To comment when data has already been computed
#         for no, (sub1, sub2) in enumerate(zip(G1, G2)):          
#             print(no, sub1, sub2)
#             G1_data_alpha[:,:,no] = hilb(sub1, 7, 13)
#             G2_data_alpha[:,:,no] = hilb(sub2, 7, 13)
#             np.savez('Alpha_V1V5_%s_%s' %(name1, name2), G1_data_alpha, G2_data_alpha)
#         print('DONE ALPHA')
######### SOMETHING TO CORRECT HERE, IT DOES NOT ENTER CORRECTLY IN THE 2ND FOR LOOP. FOR SOME REASON IT TAKES THE VALUES CONVERTED TO COMPLEX FROM THE 1ST LOOP AS INPUTS
#         for n, (s1, s2) in enumerate(zip(G1, G2)):          
#             print(n, s1, s2)        
#             G1_data_gamma[:,:,n] = hilb(s1, 30, 45)
#             G2_data_gamma[:,:,n] = hilb(s2, 30, 45) 
#             np.savez('Gamma_V1V5_%s_%s' %(name1, name2), G1_data_gamma, G2_data_gamma)

        # Reading subject's data after Hilbert
        # Real path to find the individual files
        #    '../../../Documents/Sources_July19/TF_Power/Data/G4/
        Alpha_V1V5 = np.load('Alpha_V1V5_%s_%s.npz' %(name1, name2))
        Gamma_V1V5 = np.load('Gamma_V1V5_%s_%s.npz' %(name1, name2))
        G1_alpha = Alpha_V1V5['arr_0']
        G2_alpha = Alpha_V1V5['arr_1']
        G1_gamma = Gamma_V1V5['arr_0']
        G2_gamma = Gamma_V1V5['arr_1']
#         print(G1_gamma, G2_gamma)

        # Averaging over subjects 
        G1_V1_alpha_avg = np.mean(G1_alpha[0,:,:], axis=1)
        G1_V5_alpha_avg = np.mean(G1_alpha[1,:,:], axis=1)
        G2_V1_alpha_avg = np.mean(G2_alpha[0,:,:], axis=1)
        G2_V5_alpha_avg = np.mean(G2_alpha[1,:,:], axis=1)
        G1_V1_gamma_avg = np.mean(G1_gamma[0,:,:], axis=1)
        G1_V5_gamma_avg = np.mean(G1_gamma[1,:,:], axis=1)
        G2_V1_gamma_avg = np.mean(G2_gamma[0,:,:], axis=1)
        G2_V5_gamma_avg = np.mean(G2_gamma[1,:,:], axis=1)
        print(G1_V5_alpha_avg.shape)
        print(G1_V5_alpha_avg.dtype)
#         print(G1_amp_V5_avg)

        # Plotting    
        fig1 = plt.figure(figsize=(18.0, 15.0), tight_layout=True)

        ax1 = fig1.add_subplot(2,2,1)
        plt.plot(timex[350:550],(1e16*G1_V1_alpha_avg[350:550].real), 'r', linestyle='dashed', label='Alpha')
#         plt.plot(timex[350:550],-(1e16*G1_V1_alpha_avg[350:550].real), 'b', label='Alpha Phase')
#         plt.plot(timex[350:550],np.abs(1e16*G1_V1_alpha_avg[350:550]), 'black', label='Alpha')
#         plt.plot(timex[350:550],-np.abs(1e16*G1_V1_alpha_avg[350:550]), 'black', label='-Alpha')
        plt.plot(timex[350:550],1e16*G1_V5_gamma_avg[350:550].real, 'b', label='Gamma')
        ax1.set_ylim([-2.5,2.5])
        plt.legend(prop={'size':24, 'weight':'bold'})
#         ax1.set_title('PRE: Phase Alpha V1 - Amplitude Gamma V5')
        ax1.set_xlabel('Time (s)')
              
        ax2 = fig1.add_subplot(2,2,2)
#         plt.plot(timex[350:550],np.angle(G2_V1_alpha_avg[350:550]), '-r', label='Alpha Phase')
        plt.plot(timex[350:550],np.abs(1e16*G2_V1_alpha_avg[350:550]), 'black', label='abs(Alpha)')
        plt.plot(timex[350:550],-np.abs(1e16*G2_V1_alpha_avg[350:550]), 'black', label='-abs(Alpha)')
        plt.plot(timex[350:550],1e16*G2_V5_gamma_avg[350:550].real, 'b', label='Gamma')
        ax2.set_ylim([-2.5,2.5])
        plt.legend(prop={'size':24, 'weight':'bold'})
#         ax2.set_title('POST: Phase Alpha V1 - Amplitude Gamma V5')
        ax2.set_xlabel('Time (s)')
        
        ax3 = fig1.add_subplot(2,2,3)
#         plt.plot(timex[350:550],np.angle(G1_V5_alpha_avg[350:550]), '-r', label='Alpha Phase')
        CS7 = plt.contourf([timex[350:550], np.angle(G2_V1_alpha_avg[350:550])], cmap='RdBu_r', extent=[-0.1, 0.7, 2, -2])
        cb7 = fig1.colorbar(CS7, ax=ax3, ticks=[-np.pi, 0, np.pi])
#         plt.plot(timex[350:550],np.abs(1e16*G1_V5_alpha_avg[350:550]), 'black', label='Alpha')
#         plt.plot(timex[350:550],-np.abs(1e16*G1_V5_alpha_avg[350:550]), 'black', label='-Alpha')
        plt.plot(timex[350:550], 1e16*G1_V1_gamma_avg[350:550].real, 'b')
        ax3.set_ylim([-2.5,2.5])
        plt.legend(['Gamma', 'ang(Alpha)'], prop={'size':24, 'weight':'bold'})
#         ax3.set_title('PRE: Amplitude Gamma V1 - Phase Alpha V5')
        ax3.set_xlabel('Time (s)')

        ax4 = fig1.add_subplot(2,2,4)
#         plt.plot(timex[350:550],np.angle(G2_V5_alpha_avg[350:550]), '-r', label='Alpha Phase')
#         plt.plot(timex[350:550],np.abs(1e16*G2_V5_alpha_avg[350:550]), 'black', label='Alpha')
#         plt.plot(timex[350:550],-np.abs(1e16*G2_V5_alpha_avg[350:550]), 'black', label='-Alpha')
#         plt.plot(timex[350:550],1e16*G2_V1_gamma_avg[350:550].real, 'b', label='Gamma')
        CS8 = plt.contourf([timex[350:550], np.angle(G2_V5_alpha_avg[350:550])], cmap='RdBu_r', extent=[-0.1, 0.7, 2, -2])
        cb8 = fig1.colorbar(CS8, ax=ax4, ticks=[-np.pi, 0, np.pi])
        plt.plot(timex[350:550], 1e16*(np.abs(G2_V1_gamma_avg[350:550])), 'b')
        ax4.set_ylim([-2.5,2.5])
        plt.legend(['abs(Gamma)', 'ang(Alpha)'], prop={'size':24, 'weight':'bold'})
#         ax4.set_title('POST: Amplitude Gamma V1 - Phase Alpha V5')
        ax4.set_xlabel('Time (s)')
        
        fig1.savefig('HilMod_%s_%s' %(name1, name2))
        
        fig2 = plt.figure(figsize=(18, 15), tight_layout=True)
        ax5 = fig2.add_subplot(2,2,1)
        CS5 = plt.contourf([timex[350:550], np.angle(G1_V1_alpha_avg[350:550])], cmap='RdBu_r', extent=[-0.1, 0.7, 0, 1])
        cb5 = fig2.colorbar(CS5, ax=ax5, ticks=[-np.pi, 0, np.pi])
        plt.plot(timex[350:550], 5e15*(np.abs(G1_V5_gamma_avg[350:550])), 'b')
        ax5.set_title('PRE: Phase Alpha V1 - Amplitude Gamma V5')
        ax5.set_xlabel('Time (s)')
        ax6 = fig2.add_subplot(2,2,3)
        CS6 = plt.contourf([timex[350:550], np.angle(G1_V5_alpha_avg[350:550])], cmap='RdBu_r', extent=[-0.1, 0.7, 0, 1])
        cb6 = fig2.colorbar(CS6, ax=ax6, ticks=[-np.pi, 0, np.pi])
        plt.plot(timex[350:550], 5e15*(np.abs(G1_V1_gamma_avg[350:550])), 'b')
        ax6.set_title('PRE: Amplitude Gamma V1 - Phase Alpha V5')
        ax6.set_xlabel('Time (s)')
              
        ax7 = fig2.add_subplot(2,2,2)
        CS7 = plt.contourf([timex[350:550], np.angle(G2_V1_alpha_avg[350:550])], cmap='RdBu_r', extent=[-0.1, 0.7, 0, 1])
        cb7 = fig2.colorbar(CS7, ax=ax7, ticks=[-np.pi, 0, np.pi])
        plt.plot(timex[350:550], 5e15*(np.abs(G2_V5_gamma_avg[350:550])), 'b')
        ax7.set_title('POST: Phase Alpha V1 - Amplitude Gamma V5')
        ax7.set_xlabel('Time (s)')
        ax8 = fig2.add_subplot(2,2,4)
        CS8 = plt.contourf([timex[350:550], np.angle(G2_V5_alpha_avg[350:550])], cmap='RdBu_r', extent=[-0.1, 0.7, 0, 1])
        cb8 = fig2.colorbar(CS8, ax=ax8, ticks=[-np.pi, 0, np.pi])
        plt.plot(timex[350:550], 5e15*(np.abs(G2_V1_gamma_avg[350:550])), 'b')
        ax8.set_title('POST: Amplitude Gamma V1 - Phase Alpha V5')
        ax8.set_xlabel('Time (s)')

        fig2.savefig('Mod_%s_%s' %(name1, name2))
            

    # Plotting  TF Power (ROI sources)
    if mode == 'tf_power':

#          # Looping over subjects to get data --> To comment when data has already been computed
        for no, (sub1, sub2) in enumerate(zip(G1, G2)):
            SourcesReco(sub1, type_tr1, name1, no+1, mode)
            SourcesReco(sub2, type_tr2, name2, no+1, mode) 
            
        # Reading subject's data in Src.Space
        for a in range(len(G1)):
#           Real path to find the individual files
#             '../../../Documents/Sources_July19/TF_Power/Data/G4/
            G1_tf_pwr[a,:,:] = np.load('Pwr_%s_%s.npy' %(name1, a+1))
            G2_tf_pwr[a,:,:] = np.load('../../../Documents/Sources_July19/TF_Power/Data/G5/Pwr_%s_%s.npy' %(name2, a+1))
        
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
        CS_1 = plt.contourf(timex[350:550], frex, G1_avg[:,350:550], cmap='RdBu_r', levels=levels, norm=colors.Normalize(vmin=vmin, vmax=vmax), extend='both')
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
        
#         # Looping over subjects to get data
        for no, (sub1, sub2) in enumerate(zip(G1, G2)):
#             SourcesReco(sub1, type_tr1, name1, no+1, mode)
            SourcesReco(sub2, type_tr2, name2, no+1, mode) 
            
        # Reading subject's data in Src.Space
        for a in range(len(G1)):
#           Real path to find the individual files
#             '../../../Documents/Sources_August19/Connectivity/Coherence/
#             G1_con[:,:,:,:,a] = np.load('ImCoh_%s_%s.npy' %(name1, a+1))
#             G2_con[:,:,:,:,a] = np.load('ImCoh_%s_%s.npy' %(name2, a+1)) 
        # New Vectors to be filled when computing Phase Slope Index (4 Averaged Frex Bands instead of 40 Frex bins)
            G1_cau[:,:,:,:,a] = np.load('Psi_%s_%s.npy' %(name1, a+1))
            G2_cau[:,:,:,:,a] = np.load('Psi_%s_%s.npy' %(name2, a+1)) 
        
#         print(G1_con.shape) # ([2,2,40,750,15]) #Connectivity between 2 labels it is always found in the position [1,0,:,:,:]
        print(G1_cau.shape) # ([2,2,4,750,15]) # Upper positive matrix is filled PSI
#         print(type(G1_con))
#         print(G1_con.dtype)
            
        # Averaging Conn over subjects 
#         G1_con_avg = np.mean(G1_con, axis=4)
#         G2_con_avg = np.mean(G2_con, axis=4)
        # Change for PSI - Average over subjects
        G1_con_avg = np.mean(G1_cau, axis=4)
        G2_con_avg = np.mean(G2_cau, axis=4)
        
        # Plotting
        min_time = -1.5
        max_time = 1.5
        num_time = 750
        timex = np.linspace(min_time, max_time, num_time)

        min_freq = 2
        max_freq = 42
        num_freq = 40
        frex = np.linspace(min_freq, max_freq, num_freq)
               
        def plot_conn(mat, timex, name, band):
            names = ['rh.V1', 'rh.V2', 'rh.MT', 'lh.V1', 'lh.V2', 'lh.MT']
            n_rows, n_cols = mat.shape[:2]
            fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True, figsize=(20.0, 20.0))
            for i in range(n_rows):
                for j in range(i + 1):
                    axes[i,j].tick_params(labelsize=20)
                    if i == j:
                        axes[i, j].set_axis_off()
                        continue

                    axes[i, j].plot(timex, mat[i, j, :].T)
                    axes[j, i].plot(timex, mat[i, j, :].T)

                    if j == 0:
                        axes[i, j].set_ylabel(names[i], fontsize=20)
                        axes[0, i].set_title(names[i], fontsize=20)
                    if i == (n_rows - 1):
                        axes[i, j].set_xlabel(names[j], fontsize=20)
                    axes[i, j].set(xlim=[-1.3, 1.3], ylim=[-0.1, 0.6])
                    axes[j, i].set(xlim=[-1.3, 1.3], ylim=[-0.1, 0.6])

                    axes[i, j].axvline(0, color='r') # Show Stim Onset
                    axes[j, i].axvline(0, color='r') # Show Stim Onset
                    
            #plt.show()
            fig.savefig('Src_Plv_%s_%s' %(band,name))
            
#         plot_conn(np.mean(G1_con_avg[:,:,5:11,:], axis=2), timex, name1, 'alpha')
#         plot_conn(np.mean(G2_con_avg[:,:,5:11,:], axis=2), timex, name2, 'alpha')
#         plot_conn(np.mean(G1_con_avg[:,:,29:40,:], axis=2), timex, name1, 'gamma')
#         plot_conn(np.mean(G2_con_avg[:,:,29:40,:], axis=2), timex, name2, 'gamma')
            
        def plot_tf (G1, G2, timex, frex, name):
            
            diffmap = np.subtract(G2,G1)
           
            vmax=0.010
            vmin=-0.010
            levels = MaxNLocator(nbins=40).tick_values(vmin, vmax)
            levels1 = MaxNLocator(nbins=40).tick_values(-0.01, 0.01)
            
            # To use when plotting PSI -> Change variables in contourf : Otherwise change for 'frex'
            frex_cau = np.array([2,7,30,45])

            fig1 = plt.figure(figsize=(15.0, 13.0))#figsize=(27.0, 3.0))

            ax1 = fig1.add_subplot(3,1,1)
            CS_1 = plt.contourf(timex[350:550], frex_cau, G1[:,350:550], cmap='RdBu_r', levels=levels, extend='both')
            cbar_1 = fig1.colorbar(CS_1, ax=ax1)
            ax1.set_title('Pre')
            #ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Frequency (Hz)')

            ax2 = fig1.add_subplot(3,1,2)
            CS_2 = plt.contourf(timex[350:550], frex_cau, G2[:,350:550], cmap='RdBu_r', levels=levels, extend='both')
            cbar_2 = fig1.colorbar(CS_2, ax=ax2)
            ax2.set_title('Post')
            #ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Frequency (Hz)')

            ax3 = fig1.add_subplot(3,1,3)
            CS_3 = plt.contourf(timex[350:550], frex_cau, diffmap[:,350:550], cmap='RdBu_r', levels=levels1, extend='both')
            cbar = fig1.colorbar(CS_3, ax=ax3)
            ax3.set_title('Difference Post - Pre')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Frequency (Hz)')

#             plt.show()
            fig1.savefig('%s' %name)
        
        for b in range(2):
            for c in range(b):
                plot_tf(G1_con_avg[b,c,:,:], G2_con_avg[b,c,:,:], timex, frex, 'Psi_Src_%s_%s_%s%s' %(name1, name2, b,c))

        def plot_avg(G1, G2, frex, name):

            fig1 = plt.figure(figsize=(15.0, 13.0))
            ax1 = fig1.add_subplot(3,1,1)
            plt.plot(frex, G1[2,0,:], label='Pre')
            plt.plot(frex, G2[2,0,:], label='Pos')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=2)
            plt.tight_layout(h_pad=2.0)
            plt.ylim([0.03,0.35])
            ax1.set_title('V1 R.H.- MT R.H.')
            ax1.set_ylabel('PLV')
            ax2 = fig1.add_subplot(3,1,2)
            plt.plot(frex, G1[3,0,:], label='Pre')
            plt.plot(frex, G2[3,0,:], label='Pos')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=2)
            plt.tight_layout(h_pad=2.0)
            plt.ylim([0.03,0.35])
            ax2.set_title('V1 R.H.- V1 L.H.')
            ax2.set_ylabel('PLV')
            ax3 = fig1.add_subplot(3,1,3)
            plt.plot(frex, G1[5,2,:], label='Pre')
            plt.plot(frex, G2[5,2,:], label='Pos')
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0, ncol=2)
            plt.tight_layout(h_pad=2.0)
            plt.ylim([0.03,0.35])
            ax3.set_title('Mt R.H.- MT L.H.')
            ax3.set_ylabel('PLV')
            ax3.set_xlabel('Frequency (Hz)')
    #         plt.show()
            fig1.savefig('%s' %name)
            
#         plot_avg(np.mean(G1_con_avg[:,:,:,350:500],3), np.mean(G2_con_avg[:,:,:,350:500],3), frex, 'PLV_avg_Src_G5_P1-Bsl')


    # Cross-Frequency analysis
    if mode == 'cross_freq':
        
        # Looping over subjects to get data
#         for no, (sub1, sub2) in enumerate(zip(G1, G2)):
#             SourcesReco(sub1, type_tr1, name1, no+1, mode)
#             SourcesReco(sub2, type_tr2, name2, no+1, mode) 
            
        # Reading subject's data in Src.Space
        for a in range(len(G1)):
#           Real path to find the individual files
#             '../../../Documents/10.19/CrossFr_Sources/Data/zPAC
            G1_cf[:,:,:,a] = np.load('../../../Documents/1st_2nd_Results/Sources/CrossFr_Src/Data/zPAC/P30/zPAC_V1V5_%s_%s.npy' %(name1, a+1))
            G2_cf[:,:,:,a] = np.load('../../../Documents/1st_2nd_Results/Sources/CrossFr_Src/Data/zPAC/P30/zPAC_V1V5_%s_%s.npy' %(name2, a+1)) 
        
        print(G1_cf.shape) # ([5,6,2,15])
        
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
        
            diffmap = np.subtract(G2,G1)
#             diffmap = np.divide(G2,G1)

            vmax=3.0
            vmin=0.25
            levels = MaxNLocator(nbins=60).tick_values(vmin, vmax)
            levels1 = MaxNLocator(nbins=60).tick_values(-1, 1) #Difference
#             levels1 = MaxNLocator(nbins=60).tick_values(0.25, 3.0) #Ratio

            fig1 = plt.figure(figsize=(15.0,4.0))

            ax1 = fig1.add_subplot(1,3,1)
            CS_1 = plt.contourf(x, y, G1, cmap='RdBu_r', levels=levels, extend='both')
            cbar_1 = fig1.colorbar(CS_1, ax=ax1)
            ax1.set_title('Pre')
            ax1.set_xlabel('Frequency for Phase(Hz)')
            ax1.set_ylabel('Freq. for Amplitude(Hz)')

            ax2 = fig1.add_subplot(1,3,2)
            CS_2 = plt.contourf(x, y, G2, cmap='RdBu_r', levels=levels, extend='both')
            cbar_2 = fig1.colorbar(CS_2, ax=ax2)
            ax2.set_title('Post')
            ax2.set_xlabel('Frequency for Phase(Hz)')
#             ax2.set_ylabel('Freq. for Amplitude(Hz)')

            ax3 = fig1.add_subplot(1,3,3)
            CS_3 = plt.contourf(x, y, diffmap, cmap='RdBu_r', extend='both', levels=levels1)
            cbar = fig1.colorbar(CS_3, ax=ax3)
            ax3.set_title('Diff Pos-Pre')
#             ax3.set_title('Ratio Pos/Pre')
            ax3.set_xlabel('Frequency for Phase(Hz)')
#             ax3.set_ylabel('Freq. for Amplitude(Hz)')

            plt.show()
            fig1.savefig('%s' %name)

        plot_tf(G1_V1pV5a_avg, G2_V1pV5a_avg, frex_lo, frex_hi, 'CF_V1pV5a_G3-G4_P3')
        plot_tf(G1_V1aV5p_avg, G2_V1aV5p_avg, frex_lo, frex_hi, 'CF_V1aV5p_G3-G4_P3')
        
        
    # Cross-Correlation Analysis
    if mode == 'cross_corr':
        
        def CrossCor(file, lo, hi):

            sf = 250 
            low = lo / sf/2
            high = hi /sf/2
            b, a = signal.butter(3, [low, high], btype='band') # Calculate coefficients
            filtered = signal.lfilter(b, a, file)# Filter signal

            nData = filtered.shape[0] * filtered.shape[2]
            conc1 = np.reshape(filtered[:,1,:], (nData))
            conc2 = np.reshape(filtered[:,1,:], (nData))
            corr = signal.correlate(conc1,conc2, mode='same', method='fft')

            mid = int(len(corr.T)/2)
            
            return corr.T[mid-125:mid+125]
        
        # Reading subject's data in Src.Space
        for a in range(len(G1)):
#           Real path to find the individual files
#             '../../../Documents/10.19/CrossFr_Sources/Data/
            G1_cc[:,a] = CrossCor(np.load('../../../Documents/10.19/CrossFr_Sources/Data/Src/Src_V1V5_%s_%s.npy' %(name1, a+1)),7,13)
            G2_cc[:,a] = CrossCor(np.load('../../../Documents/10.19/CrossFr_Sources/Data/Src/Src_V1V5_%s_%s.npy' %(name2, a+1)),7,13)
        
        # Averaging CC over subjects 
        G1_cc_avg = np.mean(G1_cc, axis=1)
        G2_cc_avg = np.mean(G2_cc, axis=1)
        
        #Plotting
        time = np.linspace(-0.5, 0.5, 250)
        fig1 = plt.figure(figsize=(11.0,7.0))
    
    #     stdev1 = np.std(G1_cc_avg)
        plt.plot(time, G1_cc_avg, label='Pre')
        plt.tick_params(labelsize=20)
    #     plt.fill_between(time[350:550], G1_cc_avg[350:550]+stdev1, G1_cc_avg[350:550]-stdev1, alpha=.1)
    #     stdev2 = np.std(G1_cc_avg)
        plt.plot(time, G2_cc_avg, label='Pos')
        plt.tick_params(labelsize=20)
    #     plt.fill_between(time[350:550], G2_cc_avg[350:550]+stdev1, G2_cc_avg[350:550]-stdev1, alpha=.1)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
        plt.ylabel('Alpha Cross Correlation - V1V5', fontsize=20)
        plt.xlabel('Time(s)', fontsize=20)
    #     plt.ylim([-2.5e-28,2.5e-28])
        plt.show()
        fig1.savefig('CC_V1V5_Alpha_%s-Bslxxx' %name2)