import numpy as np
import scipy as sp
import mne
import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy import fft, ifft, arange, stats, signal
import matlab.engine
import matlab
import itertools

def GrangerCaus (raw_path, type_tr, chans_x, chans_y):
# raw_path: Path where to find the raw data (.vhdr)
# type_tr: Correct='stim_cor' or Incorrect='stim_inc'
# chans: Channels to be averaged in the TimeFreq response
    
    # Get data as a numpy array --> trials x channels x time-points
    data = raw_path[type_tr].get_data()
    print(data.shape)

    # Dealing with channels
    channels = raw_path.info['ch_names']     
    def chan_groups(chans):      
        ch_index = np.zeros((len(chans),1))
        i = 0
        for chan2use in chans:
            #print(chan2use)
            # Retrieve indexes from channels of interest 
            for idx, chan in enumerate(channels):
                if chan == chan2use:
                    ch_index[i,0] = int(idx)
                else:
                    ch_index = ch_index
            i=i+1
        #print(ch_index)
        # Looping over ch. interest and averaging the signals
        all_ch = np.empty((data.shape[0], len(chans), data.shape[2]))
        for ch_idx, ch in enumerate(ch_index):
            all_ch[:,ch_idx,:] = data[:,int(ch),:]
        return (all_ch)
    
    ch_x = chan_groups(chans_x)
    ch_y = chan_groups(chans_y)
    pair = np.empty((data.shape[0],2,data.shape[2]))    
    pair[:,0,:] = np.mean(ch_x, axis=1)
    pair[:,1,:] = np.mean(ch_y, axis=1)

    # Prediction parameters
    trialdur = 3000
    timewin = 200 #ms
    order   = 27

    # Time windows to evaluate
    times2 = np.linspace(-100, 700, num=32, endpoint=True)#ms
    times2save = np.empty((1,times2.size))
    times2save[0,:] = times2
    #print(times2save.shape)

    # Frequency Parameters
    min_freq = 2
    max_freq = 42
    num_freq = 40
    frex = np.linspace(min_freq, max_freq, num_freq)
    order_points = 15

    # Parameters to indices
    timewin_points = round(timewin/(1000/250))
    order_points   = round(order/(1000/250))

    # Subtract mean = Detrend the ERP 
    avgelec = np.mean(pair, 0)
    elecs = np.asarray([(y - avgelec) for y in pair])
    #print(elecs.shape)

    # Convert requested times to indices
    times2saveidx = np.empty(times2save.shape)
    for t, tim in enumerate(times2save[0,:]):
        times2saveidx[0,t] = math.ceil((((trialdur/2)+tim)*(750))/trialdur)

    # Initialize
    x2y = np.zeros((1, times2save.shape[1]))
    y2x = np.zeros((1, times2save.shape[1]))
    bic = np.empty((times2save.shape[1], 15))

    tf_granger = np.zeros((2,len(frex),times2save.shape[1]))

    for i, timep in enumerate(times2save[0,:]):                 
        print(i)
        #data from all trials in this time window --> trials x channels x time-points
        a = times2saveidx[0,i]-np.floor(timewin_points/2)
        b = times2saveidx[0,i]+np.floor(timewin_points/2)-np.mod(timewin_points+1,2)
        temp = np.squeeze(elecs[:,:,int(a):int(b+1)])

        # Zscore and detrend
        for tr in range(temp.shape[1]):
            temp[tr,0,:] = sp.stats.zscore(sp.signal.detrend(np.squeeze(temp[tr,0,:])));
            temp[tr,1,:] = sp.stats.zscore(sp.signal.detrend(np.squeeze(temp[tr,1,:])));

        # Reshape
        temp = np.reshape(temp, (2, timewin_points*pair.shape[0]))

        # Auto-Regressive models calculated by armorf.m 
        eng = matlab.engine.start_matlab()
        Ax,Ex = eng.armorf(matlab.double(temp[0,:].tolist()),pair.shape[0],timewin_points-1,order_points, nargout=2)
        Ay,Ey = eng.armorf(matlab.double(temp[1,:].tolist()),pair.shape[0],timewin_points-1,order_points, nargout=2)
        Axy,E = eng.armorf(matlab.double(temp.tolist()),pair.shape[0],timewin_points-1,order_points, nargout=2)
        Ax = np.asarray(Ax)
        Ex = np.asarray(Ex)
        Ay = np.asarray(Ay)
        Ey = np.asarray(Ey)
        Axy = np.asarray(Axy)
        E = np.asarray(E)

        # G-causality Time-Domain
        y2x[0,i] = math.log(Ex/E[0,0])
        x2y[0,i] = math.log(Ey/E[1,1])

        # G-causality Freq-Domain
        eyx = E[1,1] - (E[0,1]**2 / E[0,0])
        exy = E[0,0] - (E[1,0]**2 / E[1,1])
        N = E.shape[0]

        for f, freq in enumerate(frex):
            H = np.eye(N)

            for m in range(order_points):
                H = H + Axy[:,(m)*N+1:(m+1)*N] * np.exp(-1j*(m+1)*2*np.pi*freq/250)

            Hi = np.linalg.inv(H)
            mult = np.dot(E, Hi.T)
            S = np.linalg.solve(H, mult) /250

            # G_caus in the Freq. Domain
            tf_granger[0,f,i] = math.log( np.absolute(S[1,1])/ np.absolute(S[1,1]-(Hi[1,0]*exy*np.conj(Hi[1,0])))/ 250)
            tf_granger[1,f,i] = math.log( np.absolute(S[0,0])/ np.absolute(S[0,0]-(Hi[0,1]*eyx*np.conj(Hi[0,1])))/ 250)
                      
    return y2x, x2y, tf_granger

    
