import numpy as np
import scipy
import mne
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import fft, ifft, arange
from mne.datasets import spm_face
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs, compute_source_psd_epochs, source_band_induced_power, source_induced_power, compute_source_psd
from mne import io, combine_evoked
from mayavi import mlab
from surfer import Brain  # noqa
from mne.source_space import read_source_spaces, SourceSpaces
from mne.connectivity import spectral_connectivity, seed_target_indices, phase_slope_index
from mne.viz import circular_layout, plot_connectivity_circle

from CrossFreq import *

def SourcesReco (raw_path, type_tr, name, numb, mode):
    
    signal = raw_path

    signal.set_eeg_reference('average', projection=True)
    signal.apply_proj()

    signal = signal[type_tr]

    noise_cov = mne.compute_covariance(signal, tmin=-.300, tmax=-.100, method='shrunk')

    data_path = spm_face.data_path()                    
    subjects_dir = data_path + '/subjects'
    subject = 'spm'

    # BEM = Triangs. of interfaces between different tissues --> Freesurfer
    # Inner skull, outer skull, skin
    bem = data_path + '/subjects/spm/bem/spm-5120-5120-5120-bem-sol.fif'  

    # Src = spacing='oct6' --> 4098 sources per hemisphere + Area 24 mm^2
    src = data_path + '/subjects/spm/bem/spm-oct-6-src.fif' 
    srcx = read_source_spaces(src)

    # Computing forward solution
    forward = mne.make_forward_solution(signal.info, None, src, bem, eeg=True)
    # Accesing Np array that contains info about dipoles and sensors
    leadfield = forward['sol']['data']
    print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

    # Parameters to compute the inverse solution
    snr = 3.0
    lambda2 = 1.0 / snr ** 2
    method = 'MNE'
    inverse_operator = make_inverse_operator(signal.info, forward, noise_cov, loose=0.2, depth=0.8)
    
# loose : None | float in [0, 1]
# Value that weights the source variances of the dipole components defining the tangent space of the cortical surfaces. Requires surface- based, free orientation forward solutions.

# depth : None | float in [0, 1]
# Depth weighting coefficients. If None, no depth weighting is performed.

# fixed : bool
# Use fixed source orientations normal to the cortical mantle. If True, the loose parameter is ignored.

# https://mne-tools.github.io/0.11/generated/mne.minimum_norm.make_inverse_operator.html?highlight=make%20inverse%20operator#mne.minimum_norm.make_inverse_operator

    # Times of interest
    min_time = -1.5
    max_time = 1.5
    num_time = 750
    timex = np.linspace(min_time, max_time, num_time)

    # Frequencies of interest + Wavelet cycles for e/freq
    fmin = 2.
    fmax = 42.
    numf = 40
    sfreq = 250  # Sampling frequency
    cwt_freqs = np.linspace(fmin,fmax,numf)
    cwt_n_cycles = [j / 5 for j in cwt_freqs]
    
    if mode == 'psd_v1v5':
        
#         bands = dict(theta=[2,6], alpha=[7, 13], betha=[14,29], gamma=[30, 42])
        bands = dict(alpha=[7, 13], gamma=[30, 42])
        
        # Read some labels
        names = ['rh.V1','rh.MT']#, 'rh.V2', 'lh.V1', 'lh.V2','lh.MT']
        labels_parc = [mne.read_label(data_path + '/subjects/spm/label/%s.label' % name) for name in names]
        print(labels_parc)

        for l, label in enumerate(labels_parc):
            for b, band in enumerate(bands.items()):
                print(b, band)
                stcs = compute_source_psd_epochs(signal, inverse_operator, lambda2=lambda2, 
                                                 method='MNE', fmin=band[1][0], fmax=band[1][1],
                                                 label = label, bandwidth='hann', 
                                                 return_generator=True, verbose=True)
                # compute average PSD   
                psd_avg = 0
                for i, stc in enumerate(stcs):
                    print(i, stc)
                    psd_avg += stc.data

                psd_avg /= len(signal) # Sources x Frequencies        
                stc.data = psd_avg # overwrite the last epoch's data with the average

                if l==0: lab = "V1"
                else: lab = "V5"

                stc.save('Pwr_%s_%s_%s_%s' %(lab, b, name, numb), ftype='h5')
                
    if mode == 'psd':
        
        bands = dict(theta=[2,6], alpha=[7, 13], betha=[14,29], gamma=[30, 42])

        for b, band in enumerate(bands.items()):
            print(b, band)
            stcs = compute_source_psd_epochs(signal, inverse_operator, lambda2=lambda2, method='MNE', fmin=band[1][0],
                                             fmax=band[1][1], bandwidth='hann', return_generator=True, verbose=True)
            # compute average PSD   
            psd_avg = 0
            for i, stc in enumerate(stcs):
                print(i, stc)
                psd_avg += stc.data

            psd_avg /= len(signal) # Sources x Frequencies        
            stc.data = psd_avg # overwrite the last epoch's data with the average

            stc.save('Pwr_%s_%s_%s' %(b, name, numb), ftype='h5')
                
    if mode == 'band_power':

        ### Spectral Density

        # Compute a source estimate per frequency band all sources
        bands = dict(theta=[2,6], alpha=[7, 13], betha=[14,29], gamma=[30, 42])
        
        stcs = source_band_induced_power(signal, inverse_operator, bands, method='MNE', n_cycles=3,
                                         baseline=(-.300,-.100), baseline_mode='zscore', n_jobs=4)

        for b, stci in stcs.items():
            stci.save('Pwr_%s_%s_%s' %(b, name, numb), ftype='h5')
            
    if mode == 'tf_power':        

        names = ['rh.V1','rh.MT']#, 'rh.V2', 'lh.V1', 'lh.V2','lh.MT']
        labels_parc = [mne.read_label(data_path + '/subjects/spm/label/%s.label' % name) for name in names]
        power = np.empty((len(names), numf, num_time))
        
        for l, label in enumerate(labels_parc):

            pwr, plv = source_induced_power(signal, inverse_operator, cwt_freqs, label, method='MNE',
                                        n_cycles=cwt_n_cycles, baseline=(-.300,-.100),
                                        baseline_mode='zscore', n_jobs=4) # srcs x time x freqs
            power[l,:,:] = np.mean(pwr, axis=0)  # average over sources
            
        roi_pwr = np.mean(power, axis=0) # average over broadmann areas ROI ????
        
        np.save('Pwr_%s_%s' %(name, numb), roi_pwr)
        
        # View time-frequency plots
#         plt.imshow(20 * roi_pwr,
#                    extent=[timex[0], timex[-1], cwt_freqs[0], cwt_freqs[-1]],
#                    aspect='auto', origin='lower', vmin=0., vmax=30., cmap='RdBu_r')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Frequency (Hz)')
#         #plt.title('Power (%s)' % title)
#         plt.colorbar()
#         plt.show()
        
    if mode == 'connect':
    
        ### Connectivity: Coh/Wpli/Phase Slope Ind
        
        # Compute inverse solution on signal
        stc = apply_inverse_epochs(signal, inverse_operator, lambda2, method)#, pick_ori='vector')

        # Read some labels
        names = ['rh.V1','rh.MT']#, 'rh.V2', 'lh.V1', 'lh.V2','lh.MT']
        labels_parc = [mne.read_label(data_path + '/subjects/spm/label/%s.label' % name) for name in names]

            # Visualize cortical parcellation ROI
    #         brain = Brain('spm', 'both', 'inflated', subjects_dir=subjects_dir,
    #                       cortex='low_contrast', background='white', size=(800, 600))

    #         visual_label = [label for label in labels_parc][0] # Index=0, Right hemisphere
    #         brain.add_label(visual_label, borders=False)

            # Average the source estimates within each label of the cortical parcellation
            # and each sub structures contained in the src space

        src = inverse_operator['src']
        label_ts = mne.extract_label_time_course(stc, labels_parc, src, mode='pca_flip', 
                                                     allow_empty=True,return_generator=False) 
        
        # Connectivity between 2:42 Hz
#         con, freqs, times, n_epochs, _ = spectral_connectivity(label_ts, method='imcoh', mode='cwt_morlet', sfreq=sfreq,
#                                                             cwt_freqs=cwt_freqs, cwt_n_cycles = cwt_n_cycles,
#                                                             fmin=fmin, fmax=fmax, faverage=False, n_jobs=1)
        # Phase Slope Index Freqs. Bands
        con, bands, times, n_epochs, _ = phase_slope_index(label_ts, mode='cwt_morlet', sfreq=sfreq,
                                                            cwt_freqs=cwt_freqs, cwt_n_cycles = cwt_n_cycles,
                                                            fmin=(2,7,14,30), fmax=(6,13,29,45), n_jobs=1)

        np.save('Psi_%s_%s' %(name, numb), con)
        
    if mode == 'cross_freq':

        # Compute inverse solution on signal
        stc = apply_inverse_epochs(signal, inverse_operator, lambda2, method)#, pick_ori='vector')

        # Read some labels
        names = ['rh.V1', 'rh.MT']#, 'rh.V2', 'lh.V1', 'lh.V2','lh.MT']
        labels_parc = [mne.read_label(data_path + '/subjects/spm/label/%s.label' % name) for name in names]

        src = inverse_operator['src']
        label_ts = mne.extract_label_time_course(stc, labels_parc, src, mode='pca_flip', 
                                                             allow_empty=True,return_generator=False) 
        
        data = np.array(label_ts)
        print(data.shape)
        
        np.save('Src_V5V5_%s_%s.npy' %(name, numb), data)

        # Read already saved data
#         data = np.load('Src_V1V5_%s_%s.npy' %(name, numb))
        
        ZPAC = CrossFreq(data)
        np.save('zPAC_V5V5_%s_%s.npy' %(name, numb), ZPAC)
        
        