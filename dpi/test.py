#encoding:utf8

import os
import sys
import matplotlib.pyplot as plt
import numpy.fft as fft
import pdb
import numpy as np

from QTdata.loadQTdata import QTloader

def load_qt():
    '''Load data from QTdb.'''
    
    qt = QTloader()
    sig = qt.load('sel100')
    plt.plot(sig['sig'])
    plt.show()

def HPF(raw_sig, fs = 250.0, fc = 8.0):
    '''High pass filtering of raw_sig.'''
    padding_size = 0
    if padding_size > 0:
        raw_sig.extend([0,] * padding_size)

    freq_arr = fft.fft(raw_sig)
    # High pass filtering
    len_freq = freq_arr.size
    N = len_freq
    
    for ind in xrange(0, len_freq):
        freq_index = ind
        if ind > (N - 1.0) / 2.0:
            freq_index = N - 1 - ind
        cur_frequency = freq_index * 2 * fs / (N - 1.0);
        filter_val = 1.0
        if cur_frequency <= fc:
            filter_val = 0.5 - 0.5 * np.cos(np.pi * cur_frequency / fc)

        freq_arr[ind] *= filter_val

    rev_sig = fft.ifft(freq_arr)

    # plt.plot(rev_sig, label = 'reversed_signal')
    # plt.plot(raw_sig, label = 'original signal')
    # plt.title('Reverse signal')
    # plt.legend()
    # plt.show()

    return rev_sig

def DPI():
    '''High pass filtering.'''
    qt = QTloader()
    sig = qt.load('sel100')
    raw_sig = sig['sig'];
    fsig = HPF(raw_sig)

    # DPI
    m1 = 100
    m2 = 300
    len_sig = fsig.size
    dpi_arr = list()
    for ind in xrange(0, len_sig):
        lower_index = ind + m1 + 1
        upper_index = ind + m1 + m2

        if upper_index >= lower_index:
            s_avg = float(np.sum(np.abs(fsig[lower_index: upper_index + 1])))
            s_avg /= m2
        else:
            s_avg = 1.0
        dpi_val = np.abs(fsig[ind]) / s_avg

        dpi_val /= 5.0
        dpi_arr.append(dpi_val)

    plt.plot(dpi_arr, label = 'DPI')
    plt.plot(fsig, label = 'fsig')
    plt.title('DPI')
    plt.legend()
    plt.show()


    
DPI()
