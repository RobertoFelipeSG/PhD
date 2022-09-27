import numpy as np
import scipy
import random
from scipy import fft, ifft, arange, signal

def CrossFreq (data):
# data: Input Numpy Array [trials x chans x time-points]

    ch_index = np.array([[0],[1]])

    # Lo Frequency Parameters
    min_lo = 3
    max_lo = 15
    num_lo = 6
    frex_lo = np.linspace(min_lo, max_lo, num_lo)

    # Hi Frequency Parameters
    min_hi = 22
    max_hi = 42
    num_hi = 5
    frex_hi = np.linspace(min_hi, max_hi, num_hi)

    # Parameters Morlet Wavelet
    cyc_rng = np.array([3, 5]) #10?
    srate = 250
    timew = arange(-2,2,1/srate)
    s_lo = (np.logspace(np.log10(cyc_rng[0]), np.log10(cyc_rng[1]), num_lo)) / (2*np.pi*frex_lo)
    s_hi = (np.logspace(np.log10(cyc_rng[0]), np.log10(cyc_rng[1]), num_hi)) / (2*np.pi*frex_hi)
    half_w = (len(timew)/2) +1
    half_w = int(half_w)

    # Initialize output Time-Frequency
    ZPAC    = np.empty((len(frex_hi), len(frex_lo), len(ch_index)))#, data.shape[0], 150))#, data.shape[2]))
    power_data = np.empty((len(ch_index), len(frex_hi), data.shape[0], data.shape[2]), dtype=int)
    phase_data = np.empty((len(ch_index), len(frex_hi), data.shape[0], data.shape[2]), dtype=complex)
    nData = data.shape[0] * data.shape[2] #* len(ch_index)
    nConv = len(timew) + nData - 1

    # # Looping over channels
    for ch in range(ch_index.shape[0]):
        print(ch)

        if ch == 0:
            lo = 0
            hi = 1
        elif ch == 1:
            lo = 1
            hi = 0

        # FFT on all trials concatenated
        data_lo = np.reshape(data[:,ch_index[lo,0],:], (1,nData))
        data_hi = np.reshape(data[:,ch_index[hi,0],:], (1,nData))
        datax_lo = fft(data_lo, nConv) 
        datax_hi = fft(data_hi, nConv)
        i=1

        # Convolution for Lower Frequency Phase
        for idx1, f1 in enumerate(frex_lo):
            print(i)

            # Create best wavelet according to frequency of analysis (best resolution)
            wavelet = (np.exp(2j*np.pi*f1*timew)) * (np.exp((-timew**2)/(2*s_lo[idx1]**2)))
            wavex = fft(wavelet, nConv)
            wavex = wavex / max(wavex)
            convo = ifft(datax_lo*wavex)
            convo = convo[0, half_w-1 : len(convo)-half_w+1]

            # Reshape into trials x time
            lo_frex_phase = np.reshape(convo, (data.shape[0], data.shape[2]))

            # Looping over Frequencies High Power
            for idx2, f2 in enumerate(frex_hi):

                # Create best wavelet according to frequency of analysis (best resolution)
                wavelet = (np.exp(2j*np.pi*f2*timew)) * (np.exp((-timew**2)/(2*s_hi[idx2]**2)))
                wavex = fft(wavelet, nConv)
                wavex = wavex / max(wavex)
                convo = ifft(datax_hi*wavex)
                convo = convo[0, half_w-1 : len(convo)-half_w+1]

                # Reshape into trials x time
                hi_frex_power = np.reshape(convo, (data.shape[0], data.shape[2]))

                # Computing power 
                power_data = (np.abs(hi_frex_power))**2
                phase_data = np.angle(lo_frex_phase)

                # Calculate Oberved PAC
                ObsPAC = np.abs(np.mean(np.reshape(power_data[:,350:500],(1,-1)) * np.exp(1j*(np.reshape(phase_data[:,350:500],(1,-1))))))

                # Permutation and Z-scoring
                reps = 2000
                PermPAC = np.empty((1,reps))

                for r in range(reps):
                    randtime = [random.randint(0,data.shape[2]) for i in range(data.shape[0])]

                    # Shifting (Permuting) time intervals/windows within the same trial
                    for tr in range(data.shape[0]):
                        power_data[tr,:] = np.roll(power_data[tr,:],randtime)

                    PermPAC[0,r] = np.abs(np.mean(np.reshape(power_data[:,350:500],(1,-1)) * np.exp(1j*(np.reshape(phase_data[:,350:500],(1,-1))))))

                ZPAC[idx2,idx1,ch] = (ObsPAC - np.mean(PermPAC)) / (np.std(PermPAC))
            i = i+1
            
    return ZPAC
    