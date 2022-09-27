import numpy as np
import mne
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mne.preprocessing import ICA
from mne.datasets import sample

from time import time

def Preprocessing_Burst (sub_id, raw_path):
# sub_id: Name to be given to the saved file
# raw_path: Path where to find the raw data (.vhdr)

    # Read input file
    raw = mne.io.read_raw_brainvision(raw_path, preload=True)

#     # Read montage file
#     montage = mne.channels.read_montage(kind='easycap_64')
#     #print(montage)
#     raw.set_montage(montage, set_dig=True)

    # Read montage file
    montage = mne.channels.read_custom_montage(fname='ActCap_Sion.bvef')#, unit='auto')
    print(type(montage))
    raw.set_montage(montage)

    # Event triggers and conditions
#     events = mne.find_events(raw)
    events, _ = mne.events_from_annotations(raw)
    
    eventsofint = np.zeros((300, 3), dtype=int)
#     sha_eventsofint = np.zeros((150, 3), dtype=int)
    ver_cor = np.zeros((1,3), dtype=int)
    ver_inc = np.zeros((1,3), dtype=int)
    sha_cor = np.zeros((1,3), dtype=int)
    sha_inc = np.zeros((1,3), dtype=int)
    j=0
            
    for i, event in enumerate(events):
        if event[2,]==4 and events[i-1,2]==8 and events[i-2,2]==1:
            ver_cor[:,0:2] = events[i-1,0:2]
            ver_cor[:,2] = int(184)
            eventsofint[j,:] = ver_cor
            j += 1
        elif event[2,]==6 and events[i-1,2]==8 and events[i-2,2]==1:
            ver_inc[:,0:2] = events[i-1,0:2]
            ver_inc[:,2] = int(186)
            eventsofint[j,:] = ver_inc
            j += 1
        elif event[2,]==4 and events[i-1,2]==8 and events[i-2,2]==2:
            sha_cor[:,0:2] = events[i-1,0:2]
            sha_cor[:,2] = int(284)
            eventsofint[j,:] = sha_cor
            j += 1
        elif event[2,]==6 and events[i-1,2]==8 and events[i-2,2]==2:
            sha_inc[:,0:2] = events[i-1,0:2]
            sha_inc[:,2] = int(286)
            eventsofint[j,:] = sha_inc
            j += 1

    a = isinstance(eventsofint, int)
    print(a)
#     print(eventsofint)
    print(eventsofint.shape)    

    # Set EEG average reference
    reference = raw.set_eeg_reference(ref_channels='average', projection=False) 

    # Filtering
    raw.notch_filter(freqs=(50,100), method='spectrum_fit', filter_length='5s')
    raw.filter (l_freq=0.5, h_freq=105.0)
    raw.filter (1., None, n_jobs=1, fir_design='firwin')
 
    # Intermediate step
    #raw.plot_psd(tmax=np.inf, fmax=45)
    picks = mne.pick_types(raw.info, meg=False, eeg=True)
    print(picks)

    # Epochs definition
    baseline = (-.300, -.100)  # means from the first instant to t = 0
    tmin = -1.50
    tmax = 2.50
    event_id = dict(verum_cor=184, verum_inc=186, sham_cor=284, sham_inc=286)
    epochs = mne.Epochs(raw, eventsofint, event_id, tmin, tmax, proj=False, picks=picks, detrend=1,
                        baseline=baseline, preload=True) # Based on signals from Trigger
    print(epochs)
    print(epochs.event_id)
      
    # baseline noise cov, not a lot of samples
    #noise_cov = mne.compute_covariance(epochs, tmax=0., method='shrunk',                                   #verbose='error')
    #print(noise_cov)

    # Plot whitening process
    #evoked = epochs.average()
    #evoked.plot_white(noise_cov, time_unit='s')

    # Plot & inspect epochs
    scalings = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=5e-4,
         emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4,
         whitened=10.)
    stimuli = mne.pick_events(events, include=[1,2,3])
    color = {1: 'blue', 2: 'red'}#, 3: 'green', 4: 'c', 5: 'black', 32: 'blue'}
    epochs.plot(block=True, scalings=scalings , n_channels=64, n_epochs=1, event_colors=color, events=stimuli)#, noise_cov=noise_cov)#

    # Interpolate bad epochs
    mne.Epochs.interpolate_bads(epochs, reset_bads=False, mode='accurate', verbose=False)
    print(epochs.info['bads'])
    
    # Resampling
    epochs_resampled = epochs.copy().resample(250, npad='auto')
    print('New sampling rate:', epochs_resampled.info['sfreq'], 'Hz')

    # ICA 
    ica = ICA(method='fastica', random_state=0)
    ica.fit(epochs_resampled)
    ica.plot_sources(epochs_resampled, block=True)
    print(ica.exclude)
    preprocessed = ica.apply(epochs_resampled)

    #Save output
    preprocessed.save('%s_preprocessed_epo.fif' %sub_id)
    
    #Return output
    return preprocessed