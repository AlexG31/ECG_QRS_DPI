#encoding:utf8

import os
import sys
import matplotlib.pyplot as plt
import math
import time
import numpy.fft as fft
import pdb
import numpy as np

from QTdata.loadQTdata import QTloader

class DPI_QRS_Detector:
    '''QRS Detector with DPI.'''
    def __init__(self, debug_info = dict()):
        self.debug_info = debug_info
        pass

    def HPF(self, raw_sig, fs = 250.0, fc = 8.0):
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
    def QRS_Detection(self, raw_sig, fs = 250.0):
        '''High pass filtering.'''

        if 'time_cost' in self.debug_info:
            start_time = time.time()

        fsig = self.HPF(raw_sig)

        # DPI
        m1 = -2
        # According to maximum heart rate
        min_distance_to_current_QRS = fs * 0.3
        N_m2 = int(fs * 1.71)
        len_sig = fsig.size

        qrs_arr = list()
        ind = 10

        while ind < len_sig:
            dpi_arr = list()
            for m2 in xrange(0, N_m2):
                lower_index = ind + m1 + 1
                upper_index = ind + m1 + m2

                lower_index = max(0, lower_index)
                upper_index = min(len_sig - 1, upper_index)

                if upper_index >= lower_index:
                    s_avg = float(np.sum(np.abs(fsig[lower_index: upper_index + 1])))
                    # s_avg /= m2 ** 0.5
                    s_avg /= math.pow(m2 ,0.5)
                else:
                    s_avg = 1.0

                # Prevent 0 division error
                if s_avg < 1e-6:
                    s_avg = 1.0

                dpi_val = np.abs(fsig[ind]) / s_avg
                dpi_arr.append(dpi_val)
            # Find cross zeros
            dpi_difference = [x[1] - x[0] for x in zip(dpi_arr, dpi_arr[1:])]
            cross_zero_positions = [0,] * len(dpi_difference)
            peak_arr = list()
            valley_arr = list()
            for diff_ind in xrange(1, len(cross_zero_positions)):
                if dpi_difference[diff_ind] == 0:
                    cross_zero_positions[diff_ind] = 2
                elif dpi_difference[diff_ind - 1] * dpi_difference[diff_ind] < 0:
                    if dpi_difference[diff_ind] > 0:
                        cross_zero_positions[diff_ind] = -1
                        valley_arr.append(diff_ind)
                    else :
                        cross_zero_positions[diff_ind] = 1
                        peak_arr.append(diff_ind)

            # Find max swing
            max_swing_value = None
            max_swing_pair = [0,0]
            prev_peak_position = None
            for cross_ind, val in enumerate(cross_zero_positions):
                if val == 1:
                    prev_peak_position = cross_ind
                elif val == -1:
                    if prev_peak_position is not None:
                        cur_amplitude_difference = dpi_arr[prev_peak_position] - dpi_arr[cross_ind]
                        if cross_ind >= min_distance_to_current_QRS:
                            if max_swing_value is None or max_swing_value < cur_amplitude_difference:
                                max_swing_value = cur_amplitude_difference
                                max_swing_pair = [prev_peak_position, cross_ind]
                    prev_peak_position = None
            if max_swing_value is None:
                # Special case: a large blank region without QRS
                if ind + N_m2 < len_sig:
                    ind += N_m2
                    continue
                break
            center_pos = sum(max_swing_pair) / 2.0
            search_radius = fs * 285.0 / 1000

            search_left = int(max(0, center_pos - search_radius + ind))
            search_right = int(min(len_sig - 1, center_pos + search_radius + ind))

            
            qrs_position = int(center_pos + ind)
            max_qrs_amplitude = fsig[qrs_position]

            for sig_ind in xrange(search_left, search_right + 1):
                sig_val = fsig[sig_ind]
                if sig_val > max_qrs_amplitude:
                    max_qrs_amplitude = sig_val
                    qrs_position = sig_ind


            # debug
            if 'decision_plot' in self.debug_info and ind > 28400:
                plt.plot(xrange(ind, ind + len(dpi_arr)), dpi_arr, label = 'DPI')
                plt.plot(fsig, label = 'fsig')
                amp_list = [fsig[x] for x in qrs_arr]
                plt.plot(qrs_arr, amp_list, 'g^', markersize = 12, label = 'QRS detected')
                plt.plot(np.array(max_swing_pair) + ind, [dpi_arr[x] for x in max_swing_pair], 'md', markersize = 12,
                         label = 'max_swing_pair')
                plt.plot([search_left, search_right], [fsig[search_left], fsig[search_right]],
                         'rd', markersize = 12,
                         label = 'search region(Searching for max amplitude)')
                plt.plot(ind, fsig[ind], 'mx', markersize = 12,
                         label = 'current index position')
                plt.plot(qrs_position, fsig[qrs_position], 'ro', markersize = 12,
                         label = 'Next QRS position')
                plt.plot(np.array(peak_arr) + ind, [dpi_arr[x] for x in peak_arr],
                         'r^', markersize = 12, label = 'dpi peaks')
                plt.plot(np.array(valley_arr) + ind, [dpi_arr[x] for x in valley_arr],
                         'rv', markersize = 12, label = 'dpi valleys')
                plt.title('DPI')
                plt.legend()
                plt.show()
            
            if qrs_position >= len_sig or qrs_position <= ind:
                # print 'Breaking...'
                # print 'ind = %d, qrs_position = %d' %(ind, qrs_position)
                break
            qrs_arr.append(qrs_position)
            ind = qrs_position


                

        if 'time_cost' in self.debug_info:
            print '*' * 15
            print '** Time cost ** %f seconds' % (time.time() - start_time)
            print '*' * 15

        print 'current index:', ind

        # plt.plot(xrange(ind, ind + len(dpi_arr)), dpi_arr, label = 'DPI')
        plt.plot(fsig, label = 'fsig')
        amp_list = [fsig[x] for x in qrs_arr]
        plt.plot(qrs_arr, amp_list, 'r^', markersize = 12)
        plt.title('DPI')
        plt.legend()
        plt.show()
    

if __name__ == '__main__':
    qt = QTloader()
    recname = qt.getreclist()[8]
    print 'record name:', recname

    sig = qt.load(recname)
    raw_sig = sig['sig'][0:100000]


    detector = DPI_QRS_Detector(debug_info = {'time_cost':True,
                                              'decision_plot':True})
    detector.QRS_Detection(raw_sig)
    
