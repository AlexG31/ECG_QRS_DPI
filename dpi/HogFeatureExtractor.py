#encoding:utf-8
"""
Extrace Hog1d feature
Author : Gaopengfei
"""
import os
import sys
import json
import math
import pickle
import pywt
import bisect
import logging
import random
import time
import pdb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

from HogClass import HogClass

from QTdata.loadQTdata import QTloader

# Global logger
# log = logging.getLogger()

# HOG 1D Feature Extractor for QT database
class HogFeatureExtractor(object):
    def __init__(self, target_label = 'P'):
        '''Hog 1D feature extractor.
        Inputs:
            target_label: label to detect. eg. 'T[(onset)|(offset)]{0,1}', 'P'
        '''
        self.qt = None
        self.qt = QTloader()

        # Feature length
        self.fixed_window_length = 250

        # Training Samples.
        self.signal_segments = []
        self.training_vector = []
        self.target_biases = []

        self.target_label = target_label

        # Define the segemnt length for each individual hog
        self.hog = HogClass(segment_len = 20)

        # ML models
        self.gbdt = None

    def GetDiffFeature(self, signal_segment, diff_step = 4):
        '''Get Difference feature.'''

        hog_arr = self.hog.GetRealHogArray(signal_segment,
                                      diff_step = diff_step,
                                      debug_plot = False)
        current_feature_vector = np.array([])
        for hog_vec in hog_arr:
            current_feature_vector = np.append(current_feature_vector,
                                               hog_vec);
        return current_feature_vector

    def GetTrainingSamples(self, sig_in, expert_labels):
        '''Form Hog1D feature.'''
        # Make sure the x indexes are in ascending order.
        expert_labels.sort(key = lambda x: x[0])

        for expert_index in xrange(0, len(expert_labels)):
            pos, label = expert_labels[expert_index]
            if label != 'R':
                continue

            # Cut out the ECG segment that end with current R peak.
            signal_segment, target_bias = self.CutSegment(sig_in,
                                                     expert_labels,
                                                     expert_index,
                                                     fixed_window_length = self.fixed_window_length)
            # Skip invalid values
            if target_bias is None:
                continue
            self.signal_segments.append(signal_segment)
            self.target_biases.append(target_bias)

            # plt.plot(signal_segment)
            # plt.plot(target_bias, np.mean(signal_segment), marker = 'd', markersize = 12)
            # plt.show()

            # hog_arr = self.hog.ComputeHog(signal_segment,
                                          # diff_step = 4,
                                          # debug_plot = False)
            # # plt.plot(signal_segment)
            # # plt.grid(True)
            # # plt.show()
            # current_feature_vector = np.array([])
            # for hog_vec in hog_arr:
                # current_feature_vector = np.append(current_feature_vector,
                                                   # hog_vec);
            current_feature_vector = np.array([])
            current_feature_vector = np.append(current_feature_vector,
                                               self.GetDiffFeature(signal_segment,
                                                   diff_step = 1))
            current_feature_vector = np.append(current_feature_vector,
                                               self.GetDiffFeature(signal_segment,
                                                   diff_step = 4))
            current_feature_vector = np.append(current_feature_vector,
                                               self.GetDiffFeature(signal_segment,
                                                   diff_step = 8))

            self.training_vector.append(current_feature_vector)


    def Train(self, reclist):
        '''Training with Qt data.'''
        for rec_name in reclist:
            sig_struct = self.qt.load(rec_name)
            raw_signal = sig_struct['sig']
            
            # Expert samples from Qt database
            expert_labels = self.qt.getExpert(rec_name)

            # Collect training vectors
            self.GetTrainingSamples(raw_signal, expert_labels)
            
            # Check
            # fixed_len = len(self.training_vector[0])
            # for vec in self.training_vector:
                # if len(vec) != fixed_len:
                        # print 'Error: new len:', len(vec)
        for vec in self.training_vector:
            for val in vec:
                if isinstance(val, float) == False:
                    raise Exception('val = {}'.format(val))
        # Training GBDT models
        self.gbdt = GradientBoostingRegressor(n_estimators=100,
                learning_rate=0.1, max_depth=1, random_state=0,
                loss='ls').fit(self.training_vector, self.target_biases)

    def LoadModel(self, model_object):
        '''Load Model object.'''
        self.gbdt = model_object

    def Testing(self, sig_in, expert_labels):
        '''Testing given ECG.'''
        
        detected_positions = list()
        # debug
        # debug_count = 7
        for expert_index in xrange(0, len(expert_labels)):
            pos, label = expert_labels[expert_index]
            if label != 'R':
                continue

            # debug_count -= 1
            # if debug_count < 0:
                # break

            # Cut out the ECG segment that end with current R peak.
            signal_segment, target_bias = self.CutSegment(sig_in,
                                                     expert_labels,
                                                     expert_index,
                                                     fixed_window_length = 250)
            # Testing
            current_feature_vector = np.array([])
            current_feature_vector = np.append(current_feature_vector,
                                               self.GetDiffFeature(signal_segment,
                                                   diff_step = 1))
            current_feature_vector = np.append(current_feature_vector,
                                               self.GetDiffFeature(signal_segment,
                                                   diff_step = 4))
            current_feature_vector = np.append(current_feature_vector,
                                               self.GetDiffFeature(signal_segment,
                                                   diff_step = 8))

            current_feature_vector = current_feature_vector.reshape(1,-1)
            predict_pos = self.gbdt.predict(current_feature_vector)

            # print 'Predict position:', predict_pos

            # Display results
            local_pos = predict_pos + self.fixed_window_length - 1
            local_pos = int(local_pos)
            # plt.plot(signal_segment)
            # plt.plot(local_pos, signal_segment[local_pos], marker = 'o',
                     # markersize = 12)
            # plt.grid(True)
            # plt.title('Testing function')
            # plt.show()

            # Append the global position
            detected_positions.append(predict_pos + pos)
            
        return detected_positions

    def TestingQt(self, record_name):
        sig_struct = self.qt.load(record_name)
        sig_in = sig_struct['sig']
        expert_labels = self.qt.getExpert(record_name)

        
        # debug
        debug_count = 7
        for expert_index in xrange(0, len(expert_labels)):
            pos, label = expert_labels[expert_index]
            if label != 'R':
                continue

            debug_count -= 1
            if debug_count < 0:
                break
            # Cut out the ECG segment that end with current R peak.
            signal_segment, target_bias = self.CutSegment(sig_in,
                                                     expert_labels,
                                                     expert_index,
                                                     fixed_window_length = 250)
            # Testing
            current_feature_vector = np.array([])
            current_feature_vector = np.append(current_feature_vector,
                                               self.GetDiffFeature(signal_segment,
                                                   diff_step = 1))
            current_feature_vector = np.append(current_feature_vector,
                                               self.GetDiffFeature(signal_segment,
                                                   diff_step = 4))
            current_feature_vector = np.append(current_feature_vector,
                                               self.GetDiffFeature(signal_segment,
                                                   diff_step = 8))

            # Suppress warning
            current_feature_vector = current_feature_vector.reshape(1,-1)
            predict_pos = self.gbdt.predict(current_feature_vector)

            print 'Predict position:', predict_pos

            # Display results
            local_pos = predict_pos + self.fixed_window_length - 1
            local_pos = int(local_pos)
            plt.plot(signal_segment)
            plt.plot(local_pos, signal_segment[local_pos], marker = 'o',
                     markersize = 12)
            plt.grid(True)
            plt.title(record_name)
            plt.show()

    def CutSegment_T(self, sig_in, expert_labels, expert_index,
                   fixed_window_length = 250 * 1):
        '''Get equal length signal_segments starts at expert_index.
        Inputs:
            sig_in: Input ECG signal.
            expert_labels: Annotation list of form [(pos, label), ...]
            expert_index: The index of the element in expert_labels that
                          has label 'R'.
            fixed_window_length : return signal's length
        Returns:
            signal_segment: Cropped signal segment.
            target_bias: (May be None)The bias respect to the expert_index's
                         position.
        '''
        current_R_pos = expert_labels[expert_index][0]
        ecg_segment = np.zeros(fixed_window_length)
        left_bound = max(0, current_R_pos - fixed_window_length + 1)
        right_bound = min(current_R_pos + fixed_window_length - 1, len(sig_in) - 1)
        len_ecg_data = abs(current_R_pos - right_bound) + 1
        ecg_segment[:len_ecg_data] = np.array(
                                        sig_in[current_R_pos: current_R_pos + len_ecg_data])
        
        
        previous_R_pos = None
        next_T_pos = None
        for ind in xrange(expert_index + 1, len(expert_labels)):
            cur_pos, cur_label = expert_labels[ind]
            if cur_label == 'R':
                if previous_R_pos is None:
                    previous_R_pos = cur_pos
                else:
                    break
            if cur_label == self.target_label:
                if next_T_pos is None:
                    next_T_pos = cur_pos
                else:
                    break
            if abs(current_R_pos - cur_pos) >= fixed_window_length:
                break
        
        if next_T_pos is not None:
            if abs(current_R_pos - next_T_pos) >= fixed_window_length:
                local_next_T_pos = None
            else:
                # Bias respect to current_R_pos
                local_next_T_pos = next_T_pos - current_R_pos
        else :
            local_next_T_pos = None
        
        return ecg_segment, local_next_T_pos
        
        
    def CutSegment(self, sig_in, expert_labels, expert_index,
                   fixed_window_length = 250 * 1):
        '''Get equal length signal_segments starts or ends at expert_index.
        Inputs:
            sig_in: Input ECG signal.
            expert_labels: Annotation list of form [(pos, label), ...]
            expert_index: The index of the element in expert_labels that
                          has label 'R'.
            fixed_window_length : return signal's length
        Returns:
            signal_segment: Cropped signal segment.
            target_bias: (May be None)The bias respect to the expert_index's
                         position.
        '''
        # Search T wave
        if 'T' in self.target_label:
            return self.CutSegment_T(sig_in, expert_labels, expert_index,
                   fixed_window_length = fixed_window_length)
            
        current_R_pos = expert_labels[expert_index][0]
        ecg_segment = np.zeros(fixed_window_length)
        left_bound = max(0, current_R_pos - fixed_window_length + 1)
        len_ecg_data = current_R_pos - left_bound + 1
        ecg_segment[fixed_window_length - len_ecg_data:] = np.array(
                                        sig_in[left_bound: current_R_pos + 1])
        
        
        previous_R_pos = None
        previous_P_pos = None
        for ind in xrange(expert_index - 1, -1, -1):
            cur_pos, cur_label = expert_labels[ind]
            if cur_label == 'R' and previous_R_pos is None:
                previous_R_pos = cur_pos
            if cur_label == self.target_label and previous_P_pos is None:
                previous_P_pos = cur_pos
        
        # Eliminate previous R wave
        #
        # plt.plot(ecg_segment)
        # if previous_R_pos is not None:
            # local_previous_R_pos = previous_R_pos - current_R_pos + fixed_window_length - 1
            # if local_previous_R_pos >= 0:
                # plt.plot(fixed_window_length - (current_R_pos - previous_R_pos), np.mean(ecg_segment), marker = 'd', markersize = 12)
        # plt.show()

        if previous_P_pos is not None:
            if current_R_pos - previous_P_pos >= fixed_window_length:
                local_previous_P_pos = None
            else:
                # Bias respect to current_R_pos
                local_previous_P_pos = previous_P_pos - current_R_pos
        else :
            local_previous_P_pos = None
        
        return ecg_segment, local_previous_P_pos
        


if __name__ == '__main__':

    # Failed to detect:
    # sel46 sel17152 sel213 sel4046 sel16273
    hog = HogFeatureExtractor(target_label = 'P')
    rec_list = hog.qt.getreclist()

    testing_rec = rec_list[80:]
    training_rec = list(set(rec_list) - set(testing_rec))

    print 'Start training...'
    start_time = time.time()
    hog.Train(training_rec)
    print 'Training time cost : %d secs.' % (time.time() - start_time)

    for rec_name in testing_rec:
        hog.TestingQt(rec_name)
