"""
MIT License

Copyright (C) <2019>  Nicolas Pajusco <nicolas.pajusco@univ-lemans.fr>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import scipy.signal as sig


def Noise_sup(X,b,alpha):
    """
    Noise suppressor by spectral substraction (JMAP)
    Methode : Joint Maximum A Posteriori Spectral Amplitude and Phase Estimator

    Patrick J.Wolfe et Simon J.Godsill. « Efficient Alternatives to the Ephraim
    and Malah Suppression Rule for Audio Signal Enhancement ». In :EURASIP
    Journal on Advances in Signal Processing 2003.10 (sept. 2003)

    ----------------------------------------------------------------------------
    Input
    X : 2D numpy array, STFT of noisy signal
    b : 1D numpy array, noise reference
    alpha : float (0 to 1), coefficient of weight between the current and last window pros

    Output:
    Filter Gk
    """
    Gk = np.zeros(X.shape)
    i = (X[:,0] - b > 0)
    Gk[i,0] = (X[i,0] - b[i])/X[i,0]
    X[:,0] *= Gk[:,0]
    for k in range(1,int(len(X[0,:]) - 1)):
        i = np.logical_and(X[:,k]>0 ,  b > 0)
        Rpost = (X[i,k]**2/b[i]**2)-1
        Rpost_2 = Rpost
        Rpost_2[Rpost_2 < 0] = 0 # R_post with positive value
        Rpost[Rpost == -1] = 0.9999999999999999999 # decimal proplem
        Rprio = (1 - alpha)*(Rpost_2) + alpha*(X[i,k-1]**2/b[i]**2)
        Gk[:,k] = (Rprio + np.sqrt(Rprio**2 +2*(1+Rprio)*(Rprio/(1+Rpost))))/(2*(1+Rprio))
        X[:,k] = X[:,k] * Gk[:,k]
    return(Gk)

def welch(noise,win_size, oneside):
    """
    welch fonction for noise averaging

    ----------------------------------------------------------------------------
    Input
    noise : 2D numpy array, noise
    win_size : int, window size
    oneside : bool, oneside or not FFT

    Output:
    1D numpy array
    """
    win = sig.windows.hann(win_size,sym = False) # bien assymétrique
    Pw_win =  np.sum(win**2) # calcule de l'énergie de la fenêtre
    L = len(noise)
    N = L//win_size
    out = np.zeros(int(win_size))

    for k in range(N-1):
        x   = noise[int(k*win_size) : ( (k+1) * win_size )] * win
        X   = np.abs(np.fft.fft(x))**2/ Pw_win
        out = out*( k / (k+1) ) + ( 1/(k+1) ) * X

    if oneside == True :
        return(out[0:int(win_size/2)+1])
    else :
        return(out)


def Desbruitage(signal,sr,Pw_noise,alpha,win,over,nfft,oneside):
    """
    Main fonction for spectral substraction

    ----------------------------------------------------------------------------
    Input:
    signal : 1D numpy array,
    sr : int sample rate
    Pw_noise : 1D numpy array, reference noise
    alpha : float (0 to 1), coefficient of weight between the current and last window pros
    win : str window 'hann'
    over : int overlap
    nfft : int for zero padding
    oneside : bool oneside or not FFT

    Output:
    1D numpy array signal estimation
    signal size
    """
    out = np.zeros(signal.shape)
    _,_,Zxx = sig.stft(signal[2000:],fs = 1.0,window= sig.windows.hann(win,
        sym = False),nperseg=win,noverlap=over,nfft= nfft,return_onesided=oneside)
    dk = Noise_sup(np.abs(Zxx),Pw_noise, alpha)
    _,sigres = sig.istft(Zxx*dk,fs = 1.0,window=sig.windows.hann(win,
        sym = False),nperseg=win,noverlap=over,nfft=nfft,input_onesided=oneside)

    out[0:len(sigres)] = np.real(sigres)
    return(out,len(sigres))





if __name__ == '__main__' :
    import scipy.io.wavfile as wav
    import matplotlib.pyplot as plt

    # Simulation of noisy sine
    tmax = 10 # s
    sr = 44.1e3
    time = np.arange(0,tmax, 1/sr)
    nb_channel = 2 # stereo
    noise = np.random.randn(int(tmax*sr), nb_channel)
    noise = (noise/np.max(noise))
    signal = np.zeros((int(tmax*sr), nb_channel ))
    signal[:,0] = np.sin(2*np.pi*200*time)
    signal[:,1] = np.sin(2*np.pi*100*time)
    signal += noise

    #parametres
    alpha = 0.98
    win = 4096
    over = int(3/4*win)
    nfft = win
    oneside = False
    ## load noise
    # sr, noise_ref = wav.read('/Users/nicolasnicolas/Desktop/Test_soft/20190611-195122_595_1008.wav')
    # noise_ref = noise/2**15 # 16 bits to -1 1.
    nb_ch = len(noise_ref[0,:])

    ## estimation bruit
    noise_ref = noise[0:int(2*sr),:] # 2 first seconds of the noise

    nb_ch = len(noise_ref[0,:])
    Pw_noise = []
    for k in range(nb_ch):
        Pw_noise.append(welch(noise_ref[:,k],win,oneside=oneside))

    ## noise_supp
    # _,signal = wav.read('/Users/nicolasnicolas/Desktop/Test_soft/20190611-195122_595_1008.wav')
    # signal = signal/2**15 # 16 bits to -1 1.
    out = np.zeros(signal.shape)
    for ch in range(nb_ch):
        out[:,ch], _ = Desbruitage(signal[:,ch],sr,Pw_noise[ch],alpha,win,over,nfft,oneside)

    wav.write('denoise_signal.wav', sr, np.int16(out*2**16))
    wav.write('initial_signal.wav', sr, np.int16(signal*2**16))


    plt.figure()
    plt.plot(time, signal[:,0], label='noisy ch1')
    plt.plot(time, signal[:,1], label='noisy ch2')
    plt.plot(time, out[:,0], label='denoise ch1')
    plt.plot(time, -out[:,1], label='denoise ch2')
    plt.plot(time, np.sin(2*np.pi*200*time), label='initial ch1')
    plt.plot(time, np.sin(2*np.pi*100*time), label='initial ch2')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (-)')
    plt.legend()
    plt.xlim(4,4.01)
    plt.show()
    
