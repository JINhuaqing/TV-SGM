# contains fns to do signal preprocess
import numpy as np
from scipy import signal

def wavelet_spectrogram(sig, fs, w_set):
    """
    Compute the wavelet power spectral density (PSD) of a given signal.

    Parameters
    ----------
    sig : array_like
        Input signal.
    fs : float
        Sampling frequency of the input signal.
    w_set : float
        Wavelet set parameter, control the time resoluation, the larger the smaller time resoluation

    Returns
    -------
    freqs: array_like
        Frequency vector filtered within the desired frequency band.
    timepoints : array_like
        Time points corresponding to the PSD values.
    psd: array_like
        PSD values filtered within the desired frequency band, not in dB. To get dB, 10*log10(psd)
    """
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return b, a
    
    
    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        zi = signal.lfilter_zi(b, a) * data[0]
        y, _ = signal.lfilter(b, a, data, zi=zi)
        return y
    fmin = 0.5
    fmax = 55
    
    # Low-pass filter coefficients
    lpf = np.array([1, 2, 3, 2, 1])
    lpf = lpf/np.sum(lpf)
    
    # Frequency vector
    fvec = np.linspace(fmin,fmax,50)
    fband = np.squeeze(np.where(np.logical_and(fvec >= 2, fvec <= 45)))
    
    # Wavelet parameters
    f_45 =  fvec[fband[-1]]
    w = (2*f_45*np.pi*w_set/fs)
    widths = w*fs/ (2*fvec*np.pi)
    t = np.linspace(0,len(sig)/fs,len(sig))
    tpoints = np.arange(0,len(sig),w_set,dtype=int)
    timepoints = t[tpoints]
    
    # Filter the input signal
    sig_filtered = butter_bandpass_filter(sig,fmin,fmax,fs)
    
    # Compute the continuous wavelet transform
    cwtm = signal.cwt(sig_filtered, signal.morlet2, widths, w=w)
    psd_raw = np.abs(cwtm[:,tpoints])**2;
    psd = psd_raw.copy()
    
    # Apply low-pass filter to the PSD
    for k in range(len(tpoints)):
        psd[:,k] = np.convolve(psd_raw[:,k],lpf,'same')
    
    # Filter the PSD within the desired frequency band
    freqs = fvec[fband]
    psd = psd[fband]
    
    return freqs, timepoints, psd