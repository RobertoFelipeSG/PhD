import numpy as np
import scipy
import mne
import matplotlib
import matplotlib.pyplot as plt
from scipy import fft, ifft, arange, signal


def TimeFreqAna (raw_path, type_tr, chans):
# raw_path: Path where to find the raw data (.vhdr)
# type_tr: Correct='stim_cor' or Incorrect='stim_inc'
# chans: Channels to be averaged in the TimeFreq response

    # Get data as a numpy array --> trials x channels x time-points
    data = raw_path[type_tr].get_data()

    # Retrieve channels of interest
    chan2use = chans
    channels = raw_path.info['ch_names']
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
#     print(ch_index)
    
    # Frequency Parameters
    min_freq = 2
    max_freq = 42
    num_freq = 40
    frex = np.linspace(min_freq, max_freq, num_freq)

    # Baseline
    baseline_window = np.array([-0.3, -0.1]);

    # Baseline time into indices
    baseidx1 = int(((baseline_window[0]+1.5)*data.shape[2])/3)
    baseidx2 = int(((baseline_window[1]+1.5)*data.shape[2])/3)

    # Parameters Morlet Wavelet
    cyc_rng = np.array([3, 10])
    srate = raw_path.info['sfreq']
    timew = arange(-2,2,1/srate)
    s = (np.logspace(np.log10(cyc_rng[0]), np.log10(cyc_rng[1]), num_freq)) / (2*np.pi*frex)
    half_w = (len(timew)/2) +1
    half_w = int(half_w)

    # Initialize output Time-Frequency
    tifr = np.empty((len(frex), data.shape[2]))
    fase = np.empty((len(frex), data.shape[2]), dtype=complex)
    itpc = np.empty((len(frex), data.shape[2]))
    tifrx = np.empty((len(ch_index), len(frex), data.shape[2]), dtype=int)
    ispcx = np.empty((len(ch_index), data.shape[2], len(frex)), dtype=complex)
    itpcx = np.empty((len(ch_index), data.shape[2], len(frex)))
    mean_tf = np.empty((len(frex), data.shape[2]))
    mean_ph = np.empty((len(frex), data.shape[2]), dtype=complex)
    mean_itpc = np.empty((len(frex), data.shape[2]))
    nData = data.shape[0] * data.shape[2] #* len(ch_index)
    nConv = len(timew) + nData - 1

    # Looping over channels
    for ch in range(ch_index.shape[0]):
       
        # FFT on all trials concatenated
        alldata = np.reshape(data[:,ch_index[ch,0],:], (1,nData))
        datax = fft(alldata, nConv)

        # Looping over Frequencies
        for idx, f in enumerate(frex):

            # Create best wavelet according to frequency of analysis (best resolution)
            wavelet = (np.exp(2j*np.pi*f*timew)) * (np.exp((-timew**2)/(2*s[idx]**2)))
            wavex = fft(wavelet, nConv)
            wavex = wavex / max(wavex)

            # Applying convolution in the Freq.Domain
            convo = ifft(datax*wavex)
            convo = convo[0, half_w-1 : len(convo)-half_w+1]

            # Reshape into trials x time
            convo = np.reshape(convo, (data.shape[0], data.shape[2]) )

            # Computing power and averaging over all trials
            tifr[idx,:] = np.mean(abs(convo)**2, axis=0)
                      
            # Phase-Angle (Angle & Magnitude) for every freq + Averaged over trials
            fase[idx,:] = np.mean(np.exp(1j*np.angle(convo)),0) # --> frex x time-points  

            # Compute ITPC --> Avg. accross trials
            itpc[idx,:] = abs(np.mean(np.exp(1j*(np.angle(convo))),0))

        # Power every channel and Normalization 
        tifrx[ch,:,:] = (10*np.log10([tifr[:,i] / (np.mean(tifr[:,baseidx1:baseidx2],1)) for i in range(data.shape[2])])).T # Decibels
        for i in range(data.shape[2]):
            tifr[:,i] = tifr[:,i] - (np.mean(tifr[:,baseidx1:baseidx2],1)) 
            tifrx[ch,:,i] = tifr[:,i] / (np.std(tifr[:,baseidx1:baseidx2],1)) # Z-scores
#         for i in range(data.shape[2]):
#             tifrx[ch,:,i] = tifr[:,i]
        
        # Angle every channel and Normalization
        ispcx[ch,:,:] = ([fase[:,i] / (np.mean(fase[:,baseidx1:baseidx2],1)) for i in range(data.shape[2])]) 
        
        # Phase Coh / Conservation every channel and Normalization
        itpcx[ch,:,:] = ([itpc[:,i] / (np.mean(itpc[:,baseidx1:baseidx2],1)) for i in range(data.shape[2])])
        
     # Power Average over channels
    mean_tf[:,:] = np.mean(tifrx, axis=0)
    
    # Phase Average over channels
    mean_ph[:,:] = (np.mean(ispcx, axis=0)).T    
    
    # ITPC Average over channels
    mean_itpc[:,:] = (np.mean(itpcx, axis=0)).T
                    
    return mean_tf, mean_ph, mean_itpc
