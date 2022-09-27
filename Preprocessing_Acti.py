import numpy as np
import mne
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from mne.preprocessing import ICA
from mne.datasets import sample

from time import time

def Preprocessing_Acti (sub_id, raw_path):
# sub_id: Name to be given to the saved file
# raw_path: Path where to find the raw data (.vhdr)

    # Read input file
    raw = mne.io.read_raw_brainvision(raw_path, preload=True)

    # Read montage file
    montage = mne.channels.read_custom_montage(fname='CACS-64_NO_REF.bvef', head_size=None, coord_frame='head')
    print(type(montage))
    raw.set_montage(montage)

    # Event triggers and conditions
#     events = mne.find_events(raw)
    events, E = mne.events_from_annotations(raw, event_id={'Stimulus/S  1': 1, 'Stimulus/S  2': 2, 'Stimulus/S  3': 3})
    print(len(events),E)
    
    eventsofint = np.zeros((150, 3), dtype=int)
    cor = np.zeros((1,3), dtype=int)
    inc = np.zeros((1,3), dtype=int)
    j=0
    for i, event in enumerate(events):
        if j==150:
            break
        elif event[2,]==1 and events[i-1,2]==3:
            cor[:,0:2] = events[i-1,0:2]
            cor[:,2] = int(31)
            eventsofint[j,:] = cor
            j += 1
        elif event[2,]==2 and events[i-1,2]==3:
            inc[:,0:2] = events[i-1,0:2]
            inc[:,2] = int(32)
            eventsofint[j,:] = inc
            j += 1
    a = isinstance(eventsofint, int)
    #print(a)
    #print(eventsofint)
    print(eventsofint.shape)    

#     incevent, lag1 = mne.event.define_target_events(events, 3, 2, sfreq=raw.info['sfreq'], tmin=-.1, tmax=3.4, new_id=32, fill_na=31)

    # Set EEG average reference
    reference = raw.set_eeg_reference(ref_channels='average', projection=False) 

    # Plot Raw data in order to delete completely bad channels
    #raw.plot(block = True, n_channels=64, remove_dc=True, events=events) 
    #print(raw.info['bads'])

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
    tmax = 1.50
    event_id = dict(stim_cor=31, stim_inc=32)
    epochs = mne.Epochs(raw, eventsofint, event_id, tmin, tmax, proj=False, picks=picks, detrend=1,
                        baseline=baseline, preload=True) # Based on signals from Trigger
    print(epochs)
    print(epochs.event_id)
      
    # baseline noise cov, not a lot of samples
#     noise_cov = mne.compute_covariance(epochs, tmax=0., method='shrunk') #verbose='error')
#     print(noise_cov)

    # Plot whitening process
#     evoked = epochs.average()
#     evoked.plot_white(noise_cov, time_unit='s')

    # Plot & inspect epochs
    scalings = dict(mag=1e-12, grad=4e-11, eeg=100e-6, eog=150e-6, ecg=5e-4,
         emg=1e-3, ref_meg=1e-12, misc=1e-3, stim=1, resp=1, chpi=1e-4,
         whitened=10.)
    stimuli = mne.pick_events(events, include=[1,2,3])
    color = {1: 'blue', 2: 'red', 3: 'green'}#, 4: 'c', 5: 'black', 32: 'blue'}
    epochs.plot(block=True, scalings=scalings , n_channels=64, n_epochs=1, event_colors=color, events=stimuli)#, noise_cov=noise_cov)#

    # Interpolate bad epochs
    mne.Epochs.interpolate_bads(epochs, reset_bads=False, mode='accurate', verbose=False)
    print(epochs.info['bads'])
    
    # Resampling
    epochs_resampled = epochs.copy().resample(250, npad='auto')
    print('New sampling rate:', epochs_resampled.info['sfreq'], 'Hz')
#     epochs_resampled.plot()

    # ICA 
    ica = ICA(method='fastica', random_state=0) #'picard', n_components=40 WHEN HESSIAN MATRIX DO NOT CONVERGE
    ica.fit(epochs_resampled)
    ica.plot_sources(epochs_resampled, block=True)
    print(ica.exclude)
    preprocessed = ica.apply(epochs_resampled)
#     preprocessed.plot()
    
    #Save output
    preprocessed.save('%s_preprocessed_epo.fif' %sub_id)
    
    #Return output
    return preprocessed