import numpy as np
from scipy.fft import fft, fftfreq, fftshift

import matplotlib.pyplot as plt

class FreqAnalyzer:

    def compute_fft_binning(self, meta):
        ''' Compute a list of frequencies to use as freq domain for FFT '''
        sampling_rate = 1000 / meta['interval_between_samples'] # Hz
        x_fft = fftfreq(meta['points_per_trace'], 1 / sampling_rate) 
        x_fft = fftshift(x_fft)
        return x_fft

    def decompose_trace_frequencies(self, meta, trace, x_fft=None, lower_freq=0, upper_freq=300, y_max=2000, plot=False):
        ''' Plots the frequencies and returns the FFT of the trace 
            https://realpython.com/python-scipy-fft/#the-scipyfft-module
        '''
        if x_fft is None:
            x_fft = self.compute_fft_binning(meta)

        y_freq_transform = np.abs(fft(trace))

        # get rid of line joining last and first points
        y_freq_transform = fftshift(y_freq_transform)

        if plot:
            plt.plot(x_fft, 
                     y_freq_transform)
            plt.xlim([lower_freq, upper_freq])
            plt.ylim([0,y_max])
            plt.show()

        return y_freq_transform