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

def SourcesReco_Bursts (raw_path, type_tr, name, numb, mode):
    
    ## Read input dataset
    signal = raw_path
    signal.set_eeg_reference('average', projection=True)
    signal.apply_proj()
    sub = signal[type_tr]
       
    ## Source Reconstruction
    data_path = spm_face.data_path()
    src = data_path + '/subjects/spm/bem/spm-oct-6-src.fif'
    bem = data_path + '/subjects/spm/bem/spm-5120-5120-5120-bem-sol.fif'
    cov = mne.compute_covariance(sub, tmin=-.300, tmax=-.100, method='empirical')
    fwd = mne.make_forward_solution(sub.info, 'fsaverage', src, bem, eeg=True)
    inv = make_inverse_operator(sub.info, fwd, cov, loose=0.2, depth=0.8)
    names = ['rh.V1','rh.MT'] # Read labels of interest
    labels_parc = [mne.read_label(data_path + '/subjects/spm/label/%s.label' % name) for name in names]

    ## Compute inverse solution on signal
    stc = apply_inverse_epochs(sub, inv, lambda2=1/3**2, pick_ori='normal', method='MNE')
    ch_index = mne.extract_label_time_course(stc, labels_parc, inv['src'], mode='pca_flip', 
                                             allow_empty=True,return_generator=False) 
    data = np.array(ch_index) # trials x chans x timep
    # print(data.shape, data.dtype)
    # data_avg = np.mean(data, axis=0) # averaging over epochs
      
    ## Times of interest
    min_time = -1.5
    max_time = 2.5
    num_time = 1000
    timex = np.linspace(min_time, max_time, num_time)

    ## Frequencies of interest + Wavelet cycles for e/freq
    fmin = 1.
    fmax = 99.
    num_freq = 44
    sfreq = 250  # Sampling frequency
    frex = np.linspace(fmin,fmax,num_freq)
    cwt_n_cycles = [j / 5 for j in frex]
    
    if mode == 'tf_wave':
    
        # Baseline definition
        baseline_window = np.array([-0.3, -0.1]);

        # Baseline time into indices
        baseidx1 = int(((baseline_window[0]+1.5)*data.shape[2])/3)
        baseidx2 = int(((baseline_window[1]+1.5)*data.shape[2])/3)

        # Parameters Morlet Wavelet
        cyc_rng = np.array([3, 10])
        srate = 250
        timew = np.arange(-2,2,1/srate)
        s = (np.logspace(np.log10(cyc_rng[0]), np.log10(cyc_rng[1]), num_freq)) / (2*np.pi*frex)
        half_w = (len(timew)/2) +1
        half_w = int(half_w)

        # Initialize output Time-Frequency
        tifr = np.empty((len(frex), data.shape[2]))
        itpc = np.empty((len(labels_parc), len(frex), data.shape[2]))
        ispc = np.empty((len(frex), data.shape[2]))
        phase_data = np.empty((len(labels_parc),len(frex), data.shape[0], data.shape[2]))
        tifrx = np.empty((len(labels_parc), len(frex), data.shape[2]), dtype=int)
        mean_tf = np.empty((len(frex), data.shape[2]))
        nData = data.shape[0] * data.shape[2] #* len(ch_index)
        nConv = len(timew) + nData - 1
        areg = np.empty((data.shape[0], len(labels_parc), data.shape[2]))
        tw_speed = np.empty((len(frex),data.shape[0], data.shape[2]))

        # Looping over channels
        for c, _ in enumerate(labels_parc):

            # FFT on all trials concatenated
            alldata = np.reshape(data[:,c,:], (1,nData))
            areg[:,c,:] = data[:,c,:]
            datax = fft.fft(alldata, nConv)

            # Looping over Frequencies
            for idx, f in enumerate(frex):

                # Create best wavelet according to frequency of analysis (best resolution)
                wavelet = (np.exp(2j*np.pi*f*timew)) * (np.exp((-timew**2)/(2*s[idx]**2)))
                wavex = fft.fft(wavelet, nConv)
                wavex = wavex / max(wavex)

                # Applying convolution in the Freq.Domain
                convo = fft.ifft(datax*wavex)
                convo = convo[0, half_w-1 : len(convo)-half_w+1]

                # Reshape into trials x time
                convo = np.reshape(convo, (data.shape[0], data.shape[2]))

                # Average over all trials
                tifr[idx,:] = np.mean(np.abs(convo)**2, axis=0)

                # Calculate angle for every channel
                phase_data[c,idx,:,:] = np.angle(convo)          # ch x frex x trials x time-points

            # Decibel Normalization    
            tifrx[c,:,:] = (10*np.log10([tifr[:,i] / (np.mean(tifr[:,baseidx1:baseidx2],1)) for i in range(data.shape[2])])).T

        # Power Average over channels and Trials
        mean_tf[:,:] = np.mean(tifrx, axis=0)                   # frex x time-points 
        np.save('Pwr_%s_%s' %(name, numb), mean_tf)
    
        # Complex Average over Trials
        avg_tr = np.mean(np.exp(1j*phase_data),2)               # ch x frex x time-points      
        np.save('Wave_%s_%s' %(name, numb), avg_tr)
        
        # Phase Difference between Areas to calculate Speed Transmission 
        ph_diff = phase_data[1,:,:,:] - phase_data[0,:,:,:]     # frex x trials x time-points
        for e in range(data.shape[0]):
            for t in range(1000):
                tw_speed[:,e,t] = 0.01 / (250*(np.rad2deg(ph_diff[:,e,t])) / (360*2*np.pi*frex))  #1cm distance V1-V5
        tw_speed = np.mean(tw_speed,1)                          # Average over Trials
        np.save('TWSpeed_%s_%s' %(name, numb), tw_speed)
        
          
    if mode == 'tf_power':        

        names = ['rh.V1','rh.MT']#, 'rh.V2', 'lh.V1', 'lh.V2','lh.MT']
        labels_parc = [mne.read_label(data_path + '/subjects/spm/label/%s.label' % name) for name in names]
        power = np.empty((len(names), num_freq, num_time))
        
        for l, label in enumerate(labels_parc):

            pwr, plv = source_induced_power(signal, inv, frex, label, method='MNE',
                                        n_cycles=cwt_n_cycles, baseline=(-.300,-.100),
                                        baseline_mode='zscore', n_jobs=4) # srcs x time x freqs
            power[l,:,:] = np.mean(pwr, axis=0)  # average over sources
            
        roi_pwr = np.mean(power, axis=0) # average over broadmann areas ROI ????
        
        np.save('Pwr_%s_%s' %(name, numb), roi_pwr)
        
        # View time-frequency plots
#         plt.imshow(20 * roi_pwr,
#                    extent=[timex[0], timex[-1], frex[0], frex[-1]],
#                    aspect='auto', origin='lower', vmin=0., vmax=30., cmap='RdBu_r')
#         plt.xlabel('Time (s)')
#         plt.ylabel('Frequency (Hz)')
#         #plt.title('Power (%s)' % title)
#         plt.colorbar()
#         plt.show()
        
    if mode == 'connect':
    
        ### Connectivity: Coh/Wpli/Plv
        
        con, freqs, times, n_epochs, _ = spectral_connectivity(ch_index, method='wpli', mode='cwt_morlet', sfreq=sfreq,
                                                            cwt_freqs=frex, cwt_n_cycles = cwt_n_cycles,
                                                            fmin=fmin, fmax=fmax, faverage=False, n_jobs=1)
        ### Phase Slope Index Freqs. Bands
#         con, bands, times, n_epochs, _ = phase_slope_index(ch_index, mode='cwt_morlet', sfreq=sfreq,
#                                                             cwt_freqs=frex, cwt_n_cycles = cwt_n_cycles,
#                                                             fmin=(2,7,14,30), fmax=(6,13,29,45), n_jobs=1)

        np.save('WPLI_%s_%s' %(name, numb), con)
        
    if mode == 'cross_freq':

        # Read some labels
        names = ['rh.V1', 'rh.MT']#, 'rh.V2', 'lh.V1', 'lh.V2','lh.MT']
        labels_parc = [mne.read_label(data_path + '/subjects/spm/label/%s.label' % name) for name in names]

        label_ts = mne.extract_label_time_course(stc, labels_parc, inv['src'], mode='pca_flip', 
                                                             allow_empty=True,return_generator=False) 
        
        data = np.array(label_ts)
        print(data.shape)
        
        np.save('Src_V1V5_%s_%s.npy' %(name, numb), data)

        # Read already saved data
#         data = np.load('Src_V1V5_%s_%s.npy' %(name, numb))
        
        ZPAC = CrossFreq(data)
        np.save('zPAC_V1V5_%s_%s.npy' %(name, numb), ZPAC)
        
        