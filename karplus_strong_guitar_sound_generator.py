# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 18:59:41 2022

@author: eljac
"""

import numpy as np
import numba

# WOW 133x faster when jitted
@numba.jit("f8[:](f8[:], i8, i8)", nopython=True, nogil=True)
def karplus_strong_jit(wavetable, n_samples, stretch_factor=1):
    """Synthesizes a new waveform from an existing wavetable, modifies last sample by averaging."""
    wave_tab = 1*wavetable
    samples = []
    i = 0
    prev = 0
    while len(samples) < n_samples:

        r = np.random.binomial(1, 1 - 1/stretch_factor)
        if r == 0:
            wave_tab[i] =  0.5 * (wave_tab[i] + prev)

        wave_tab[i] = 0.5 * (wave_tab[i] + prev)
        samples.append(wave_tab[i])
        prev = samples[-1]
        i += 1
        i = i % wave_tab.size
    return np.array(samples)



#%% Calculating
if __name__ in '__main__':

    from scipy.fftpack import fft, ifft, fftfreq

    string = np.array("Fret, (E2), (A2), (D3), (G3), (B3), (E4)".split())
    fret = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
    freq = np.array([82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.26, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77]).reshape(6,20).transpose()

    fs = 44100
    wavetable_size = (fs / freq).astype(int)

    # Low and High pass filter frequencies
    low_filter, high_filter = 5, 8000

    # Stretch factors, lets try a mapping from 1 (for lowest freq) to 2 (for highest freq)
    def my_map(x, lst): return (x - x.min()) * (lst[1] - lst[0]) / (x.max() - x.min()) + lst[0]
    stretch_factors = my_map(freq, [1,4])

    # Fades to use: Logarithmic, 1/x, linear (same as 1/x with base=1)
    log_fade = my_map( np.log(2*fs - np.arange(2*fs) ), [0,1])
    recip_fade = my_map( 1 - ( 5e-4 )**(np.arange(2*fs)/(2*fs)), [0,1])[::-1]
    lin_fade = my_map( 1 - ( 1-1e-10 )**(np.arange(2*fs)/(2*fs)), [0,1])[::-1]

    # Matrices for all the waveforms (floats) and FFTs (complex numbers)
    samples, samples_filtered = [np.empty((*freq.shape,2*fs)) for i in [0,0]]
    ffts, ffts_filtered = [np.empty((*freq.shape,2*fs), dtype=np.complex128) for i in [0,0]]

    x_freq = fftfreq(2*fs, 1/fs)

    for i in range(freq.shape[0]):
        for j in range(freq.shape[1]):
            # Different wavetables generate different types of sound
            # A random distribution of -1 and 1 turns out to give a string sound
            #wavetable = (2 * np.random.randint(0, 2, wavetable_size[i,j]) - 1).astype(float)
            #wavetable = 2*np.random.normal(0, 0.5, wavetable_size[i,j]) - 1
            wavetable = np.sin(my_map( np.arange(wavetable_size[i,j]), [0,2*np.pi]))
            #wavetable = wavetable + np.random.normal(0, 0.1, wavetable_size[i,j]) # add gaussian noise

            samples[i,j] = karplus_strong_jit(wavetable, 2*fs, stretch_factors[i,j])
            ffts[i,j] = fft(samples[i,j])

            ffts_filtered[i,j] = 1 * ffts[i,j]
            ffts_filtered[i,j][(abs(x_freq)<low_filter) | (abs(x_freq)>high_filter)] = 0
            samples_filtered[i,j] = np.real(ifft(ffts_filtered[i,j]))


    samples_filtered_log_fade = samples_filtered * log_fade
    samples_filtered_recip_fade = samples_filtered * recip_fade

    final_samples = samples_filtered_log_fade / np.max(samples_filtered_log_fade, axis=-1)[:,:,None]


    import matplotlib.pyplot as plt
    [plt.plot(x) for x in [log_fade, recip_fade, lin_fade]], plt.show()

    rows, cols = 5,3
    fig, axes = plt.subplots(rows,cols, figsize=(16,14), constrained_layout=True)
    for i in range(rows):
        for j in range(cols):
            axes[i,j].plot(samples[4*i, 2*j], lw=0.5, c='royalblue', label="KS-raw-signal")
            axes[i,j].plot(samples_filtered[4*i, 2*j], lw=0.5, c='orange', label="KS-filtered-signal")
            axes[i,j].axhline(0, ls='--', c='k', alpha=0.69)
            axes[i,j].set_title(f"Frequency = {freq[i,j]}Hz")
            axes[i,j].legend()

    plt.show()


    #% Plotting FFTs
    fig, axes = plt.subplots(rows,cols, figsize=(16,14), constrained_layout=True)
    for i in range(rows):
        for j in range(cols):
            axes[i,j].plot(x_freq, abs(ffts[4*i, 2*j]), c='r', label="FFT-raw")
            axes[i,j].plot(x_freq, abs(ffts_filtered[4*i, 2*j]), ls='--', c='limegreen', label="FFT-filtered")
            axes[i,j].set_title(f"Frequency = {freq[i,j]}Hz")
            axes[i,j].legend()
            axes[i,j].set_xlim(-freq[i,j]*10, freq[i,j]*10)
            axes[i,j].set_ylim(-0.1*np.max(abs(ffts_filtered[4*i, 2*j])), 1.1*np.max(abs(ffts_filtered[4*i, 2*j])) )

    plt.show()


    #% Concatenate waveforms into one
    y_filt = samples_filtered.swapaxes(0,1).flatten()

    fig, (ax,ay,az) = plt.subplots(3,1, figsize=(14,9), constrained_layout=True)

    ax.plot(y_filt[:len(y_filt)//10][::10])
    ax.axhline(0, ls='--', c='r')

    y_filt_log = samples_filtered_log_fade.swapaxes(0,1).flatten()
    ay.plot(y_filt_log[:len(y_filt)//10][::10])
    ay.axhline(0, ls='--', c='r')

    y_filt_recip = samples_filtered_recip_fade.swapaxes(0,1).flatten()
    az.plot(y_filt_recip[:len(y_filt)//10][::10])
    az.axhline(0, ls='--', c='r')

    plt.show()

