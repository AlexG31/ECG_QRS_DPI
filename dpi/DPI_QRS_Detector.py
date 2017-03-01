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

    def pow2length(self, length):
        '''Smallest power of 2 larger than length.'''
        ans = 1
        if length <= 0:
            return ans
        while ans < length:
            ans *= 2
        return ans

    def HPF(self, raw_sig, fs = 250.0, fc = 8.0):
        '''High pass filtering of raw_sig.'''

        if isinstance(raw_sig, list):
            len_sig = len(raw_sig)
        elif isinstance(raw_sig, np.ndarray):
            len_sig = raw_sig.size
        else:
            raise Exception('Input signal is not a list/np.array type!')

        len_pow2 = self.pow2length(len_sig)
        
        freq_arr = fft.fft(raw_sig, len_pow2)
        # High pass filtering
        len_freq = freq_arr.size

        # Skip short signals
        if len_freq <= 2:
            return raw_sig

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

        rev_sig = fft.ifft(freq_arr, len_pow2)
        rev_sig = rev_sig[:len_sig]

        return rev_sig

    def eliminate_peak_valley_pairs(self, peak_arr, valley_arr, 
                                    min_peak_valley_distance, fs):
        '''Eliminate nearby peak-valley pairs.'''
        p1 = 0
        p2 = 0
        continuous_eliminate_count = 0
        last_eliminate_position = 0

        len_peak = len(peak_arr)
        len_valley = len(valley_arr)

        filtered_peaks = list()
        filtered_valleys = list()

        while p1 < len_peak and p2 < len_valley:
            peak_pos = peak_arr[p1]
            valley_pos = valley_arr[p2]
            
            # Continuous eliminate count
            if abs(min(peak_pos, valley_pos) - last_eliminate_position) > fs / 50.0:
                continuous_eliminate_count = 0
                
            cur_distance = abs(peak_pos - valley_pos)

            # This algorithm will not eliminate the last peak and last valley
            if (cur_distance <= min_peak_valley_distance and
                    p1 < len_peak - 1 and p2 < len_valley - 1):
                if continuous_eliminate_count >= 3:
                    continuous_eliminate_count = 0
                else:
                    p1 += 1
                    p2 += 1
                    last_eliminate_position = max(peak_pos, valley_pos)
                    continuous_eliminate_count += 1
                    continue

            if peak_pos < valley_pos:
                filtered_peaks.append(peak_pos)
                p1 += 1
            else:
                filtered_valleys.append(valley_pos)
                p2 += 1
                
        # Processing remaining peaks & valleys
        while p1 < len_peak:
            peak_pos = peak_arr[p1]
            filtered_peaks.append(peak_pos)
            p1 += 1
        while p2 < len_valley:
            valley_pos = valley_arr[p2]
            filtered_valleys.append(valley_pos)
            p2 += 1

        return (filtered_peaks, filtered_valleys)
        
    def search_for_maximum_qrs(self, ind, center_pos,
            search_left, search_right, fsig):
        '''Search for maximum amplitude QRS in [search_left, search_right].'''
        qrs_position = int(center_pos + ind)
        qrs_position = min(qrs_position, len(fsig) - 1)
        qrs_position = max(qrs_position, 0)

        max_qrs_amplitude = abs(fsig[qrs_position])

        for sig_ind in xrange(search_left, search_right + 1):
            sig_val = abs(fsig[sig_ind])
            if sig_val > max_qrs_amplitude:
                max_qrs_amplitude = sig_val
                qrs_position = sig_ind
        if fsig[qrs_position] >= 0:
            return qrs_position

        # Now search for the maximum amplitude before current qrs_position
        max_qrs_amplitude = fsig[qrs_position]
        for sig_ind in xrange(search_left, qrs_position + 1):
            sig_val = fsig[sig_ind]
            if sig_val > max_qrs_amplitude:
                max_qrs_amplitude = sig_val
                qrs_position = sig_ind
            
        return qrs_position

    def QRS_Detection(self, raw_sig_in, fs = 250.0):
        '''High pass filtering.'''
        raw_sig = raw_sig_in[:]

        if 'time_cost' in self.debug_info:
            start_time = time.time()

        if isinstance(raw_sig, list):
            len_sig = len(raw_sig)
        elif isinstance(raw_sig, np.ndarray):
            len_sig = raw_sig.size
        else:
            raise Exception('Input signal is not a list/np.array type!')

        fsig = self.HPF(raw_sig, fc = 35.0)
        fsig = fsig[:len_sig]

        # DPI
        m1 = -2
        # According to maximum heart rate
        min_distance_to_current_QRS = fs * 0.2
        # N_m2 = int(fs * 1.71)
        N_m2 = int(fs * 2.21)
        # search_radius = fs * 285.0 / 1000
        search_radius = fs / 250.0 * 10
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
            peak_arr = list()
            valley_arr = list()
            for diff_ind in xrange(1, len(dpi_difference)):
                if dpi_difference[diff_ind - 1] * dpi_difference[diff_ind] < 0:
                    if dpi_difference[diff_ind] > 0:
                        valley_arr.append(diff_ind)
                    else :
                        peak_arr.append(diff_ind)

            # Connect nearby peak-valley pairs
            min_peak_valley_distance = fs / 250.0 * 3
            peak_arr, valley_arr = self.eliminate_peak_valley_pairs(peak_arr,
                                             valley_arr, 
                                             min_peak_valley_distance,
                                             fs = fs)
            # Find max swing
            max_swing_value = None
            max_swing_pair = [0,0]
            # prev_peak_position = None
            pt_peak = 0
            pt_valley = 0
            len_peaks = len(peak_arr)
            len_valleys = len(valley_arr)
            while pt_peak < len_peaks and pt_valley < len_valleys:
                peak_pos = peak_arr[pt_peak]
                valley_pos = valley_arr[pt_valley]

                # Keep 1 peak away from current QRS
                if peak_pos < valley_pos and pt_peak > -1:
                    cur_amplitude_difference = dpi_arr[peak_pos] - dpi_arr[valley_pos]
                    if peak_pos >= min_distance_to_current_QRS:
                        if max_swing_value is None or max_swing_value < cur_amplitude_difference:
                            max_swing_value = cur_amplitude_difference
                            max_swing_pair = [peak_pos, valley_pos]
                
                if peak_pos < valley_pos:
                    pt_peak += 1
                else:
                    pt_valley += 1
            if max_swing_value is None:
                # Special case: a large blank region without QRS
                if ind + N_m2 < len_sig:
                    ind += N_m2
                    continue
                break
            center_pos = sum(max_swing_pair) / 2.0

            search_left = int(max(0, max_swing_pair[0]- search_radius + ind))
            search_right = int(min(len_sig - 1, max_swing_pair[1]+ search_radius + ind))

            qrs_position = self.search_for_maximum_qrs(ind, center_pos,
                    search_left, search_right, fsig)


            # debug
            if 'decision_plot' in self.debug_info and ind > self.debug_info['decision_plot']:
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


        # plt.plot(xrange(ind, ind + len(dpi_arr)), dpi_arr, label = 'DPI')
        if 'plot_results' in self.debug_info:
            plt.figure(1)
            plt.plot(fsig, label = 'fsig')
            amp_list = [fsig[x] for x in qrs_arr]
            plt.plot(qrs_arr, amp_list, 'r^', markersize = 12)
            plt.title('DPI')
            plt.legend()
        
            plt.figure(3)
            plt.plot(raw_sig, label = 'raw signal')
            amp_list = [raw_sig[x] for x in qrs_arr]
            plt.plot(qrs_arr, amp_list, 'r^', markersize = 12)
            plt.title('Raw signal')
            plt.legend()

        return qrs_arr

if __name__ == '__main__':
    qt = QTloader()
    recname = qt.getreclist()[67]
    print 'record name:', recname

    sig = qt.load(recname)
    raw_sig = sig['sig'][0:]


    debug_info = dict()
    debug_info['time_cost'] = True
    debug_info['plot_results'] = True
    # debug_info['decision_plot'] = 25262
    detector = DPI_QRS_Detector(debug_info = debug_info)
    qrs_arr = detector.QRS_Detection(np.array(raw_sig))

    # Plot R-R histogram
    plt.figure(2)
    diff_arr = [x[1] - x[0] for x in zip(qrs_arr, qrs_arr[1:])]
    plt.hist(diff_arr, bins = 30*6, range = (0,max(450, max(diff_arr))))
    plt.title('R-R histogram')
    plt.grid(True)
    plt.show()
    
