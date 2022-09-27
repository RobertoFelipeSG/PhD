import numpy as np
import mne

def crosscor(file, lo, hi):

    sf = 250 
    low = lo / sf/2
    high = hi /sf/2
    b, a = signal.butter(3, [low, high], btype='band') # Calculate coefficients
    filtered = signal.lfilter(b, a, file) # Filter signal

    corr = np.empty([file.shape[0], file.shape[2]])
    for n in range(file.shape[0]): 
        corr[n] = signal.correlate(filtered[n,0,:],(filtered[n,1,:]), mode='same', method='fft')

    corr_mean = np.mean(corr, axis=0)
    print(corr_mean.shape)
    
    return corr_mean

