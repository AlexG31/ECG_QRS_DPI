#encoding:utf-8
"""
ECG classification with Random Forest
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
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool

import numpy as np

## Machine learning methods
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# project homepath
# 
# curfilepath =  os.path.realpath(__file__)
# curfolderpath = os.path.dirname(curfilepath)
# projhomepath = curfolderpath
# projhomepath = os.path.dirname(projhomepath)

# configure file
# conf is a dict containing keys
# with open(os.path.join(projhomepath,'ECGconf.json'),'r') as fin:
    # conf = json.load(fin)

# sys.path.append(projhomepath)
#
# my project components
# from RFclassifier.extractfeature.Regression_Feature import Regression_Feature
from QTdata.loadQTdata import QTloader

# Global logger
log  = logging.getLogger()

## Main Scripts
EPS = 1e-6

class RegressionLearner:
    def __init__(self, target_label = 'Toffset', SaveTrainingSampleFolder = None):
        # Feature&output for learner.
        self.target_label = target_label
        self.feature_pool_ = []
        self.output_pool_ = []
        self.MaxQRSWidth = 100
        self.MaxHeartBeatWidth = 350
        self.RF_TreeNumber = 40
        self.Tree_Max_Depth = 40
        self.QRSlabels = ('R',)
        self.QRS_segment_expand_length = 30
        self.SWT_diff_feature_expand = 50
        self.SWT_diff_feature_expand_skip_gap = 1

        # Signal length to extract feature from
        self.FixedSignalLength = self.MaxHeartBeatWidth + 2 * self.QRS_segment_expand_length

    def TrainQtRecords(self, record_list):
        '''API for QTdb: training model with given record_list.'''
        QTdb = QTloader()

        training_count = 1
        # Extracting feature from each record.
        for record_name in record_list:
            sig_struct = QTdb.load(record_name)
            raw_signal = sig_struct['sig']
            expert_labels = QTdb.getExpert(record_name)
            self.AddNewTrainingSignal(raw_signal, expert_labels)
            # Logging
            log.info('Extracted features from %s' % record_name)
            print '.' * training_count, '(%d/%d)' % (training_count, len(record_list))
            training_count += 1

        # Training with feature pool
        self.training()

    def TestQtRecords(self,save_folder, reclist = []):
        '''API for QTdb: testing given record_list.'''
        QTdb = QTloader()

        print 'Testing:'
        testing_count = 1
        for record_name in reclist:
            # Logging
            log.info('Testing record %s' % record_name)
            print '.' * testing_count, '(%d/%d)' % (testing_count, len(reclist))
            testing_count += 1

            sig_struct = QTdb.load(record_name)
            expert_labels = QTdb.getExpert(record_name)
            # Test lead1
            raw_signal = sig_struct['sig']
            predict_position_list = self.testing(raw_signal, expert_labels)
            test_result = zip(predict_position_list,
                    [self.target_label,]*len(predict_position_list))
            lead_result = [record_name, test_result]
            lead_result_list = []
            lead_result_list.append(lead_result)
            # Test lead2
            raw_signal = sig_struct['sig2']
            predict_position_list = self.testing(raw_signal, expert_labels)
            test_result = zip(predict_position_list,
                    [self.target_label,]*len(predict_position_list))
            lead_result = [record_name + '_sig2', test_result]
            lead_result_list.append(lead_result)

            # Save result.
            with open(
                    os.path.join(save_folder, 'result_{}'.format(record_name)),
                    'w'
                    ) as fout:
                    json.dump(lead_result_list, fout, indent = 4)
                
    def AddNewTrainingSignal(self, rawsig, expert_marklist):
        '''Form regression feature pool.'''
        # Input with rawsig & marklist.
        # 1. remove QRS from rawsig
        # 2. Do SWT.
        # 3. a) Training of T:
        #      for each T&Toffset:
        #      if there's a QRS in front of it
        #       Extract feature & output
        #    b) Training of P:
        #      for each Ponset&P&Poffset:
        #      if there's a QRS behind it
        #       Extract feature & output
        
        # For debug
        backup_signal = rawsig[:]

        # preprocess
        # expert_marklist is [(pos,label)...]
        # sort by position.
        expert_marklist.sort(key = lambda x:x[0])
        # Set self.cDlist.
        self.nonQRS_swt(rawsig,expert_marklist)

        # a) training T labels.
        if self.target_label in ['T', 'Toffset', 'Tonset']:
            self.AddT2feature(
                    rawsig,
                    expert_marklist,
                    debug = False,
                    original_signal = backup_signal
                    )
            return None
        # b) training P labels.
        # TODO
        if self.target_label in ['P', 'Ponset', 'Poffset']:
            self.AddP2feature(
                    rawsig,
                    expert_marklist,
                    debug = False,
                    original_signal = backup_signal
                    )
            return None
            

    def GetRRSegmentPrev(self, r_pos_list, r_pos):
        '''Get R-R segment, find the first previous position before r_pos.'''
        # Will do binary search on r_pos 
        # r_pos must be sorted

        # Find the first R with position < r_pos
        next_r_pos = bisect.bisect_left(r_pos_list, r_pos)
        if (next_r_pos >= 0 and 
                next_r_pos < len(r_pos_list) and
                r_pos <= r_pos_list[next_r_pos]):
            pass
        elif next_r_pos >= len(r_pos_list):
            next_r_pos = len(r_pos_list) - 1
        else:
            # r_pos should exists in r_pos_list
            # Logging error
            log.error('r_pos %d is not a labeled R peak!' % r_pos)
            print 'r_pos %d is not a labeled R peak!' % r_pos
            pdb.set_trace()

        while (next_r_pos >= 0 and
                r_pos_list[next_r_pos] >= r_pos):
            next_r_pos -= 1

        if next_r_pos >= 0:
            next_r_pos = r_pos_list[next_r_pos]
        else:
            next_r_pos = None

        # Limit the value of r_pos & next_r_pos to get segment_range
        if (next_r_pos is None or
                abs(next_r_pos - r_pos) > self.MaxHeartBeatWidth):
            next_r_pos = r_pos - self.MaxHeartBeatWidth
        # [WARNING]: This range may exceed the limit of raw signals
        segment_range = [next_r_pos, r_pos]

        return segment_range

    def GetRRSegment(self, r_pos_list, r_pos):
        '''Get RR segment.'''
        # Will do binary search on r_pos 
        # r_pos must be sorted

        # Find the first R with position larger than r_pos
        next_r_pos = bisect.bisect_left(r_pos_list, r_pos)
        if next_r_pos < len(r_pos_list) and r_pos == r_pos_list[next_r_pos]:
            # Find next r position > r_pos
            while (next_r_pos < len(r_pos_list) and
                    r_pos_list[next_r_pos] == r_pos):
                next_r_pos += 1
        else:
            # r_pos should exists in r_pos_list
            # Logging error
            log.error('r_pos %d is not a labeled R peak!' % r_pos)
        if next_r_pos < len(r_pos_list):
            next_r_pos = r_pos_list[next_r_pos]
        else:
            next_r_pos = None

        # Limit the value of r_pos & next_r_pos to get segment_range
        if next_r_pos is None or next_r_pos - r_pos > self.MaxHeartBeatWidth:
            next_r_pos = r_pos + self.MaxHeartBeatWidth
        # [WARNING]: This range may exceed the limit of raw signals
        segment_range = [r_pos, next_r_pos]

        return segment_range

    def CropSignalBySegmentRange(self,
            signal,
            segment_range,
            FixedLength,
            limit_by_right = True):
        '''
        Copy signal segment according to RR distance, padding zeros if segment exceeds the
        signnal boundaries.
        NOTE:
            Distance between segment_ranges is not guarenteed to be same.
            This function copies the data to a new list, rather than using the original
            pointer.
            This function crop the segment if it exceeds FixedLength.
        '''
        cropped_signal = list()
        left, right = segment_range[0:2]
        if left < 0:
            cropped_signal += [0,] * abs(segment_range[0])
            left = 0
        if right >= len(signal):
            cropped_signal += signal[left:]
            cropped_signal += [0,] * (right - len(signal))
        else:
            cropped_signal += signal[left:right + 1]
        # NOTE: Do not modify the left boundary since the bias of
        #       the output is based on the left boundary
        if len(cropped_signal) > FixedLength:
            if limit_by_right:
                cropped_signal = cropped_signal[0:FixedLength]
            else:
                cropped_signal = cropped_signal[-FixedLength:len(cropped_signal)]
                

        return cropped_signal
        
    def GetPreviousLabelPosition(self,
            target_label,
            pos_list,
            label_list,
            r_pos):
        '''Get previous target label from current r_pos.'''
        # pos_list must be sorted.
        r_index = bisect.bisect_left(pos_list, r_pos)

        # Find target_pos.
        target_pos = None
        for ind in xrange(r_index - 1, 0, -1):
            pos, label = (pos_list[ind], label_list[ind])
            if pos - r_pos > self.MaxHeartBeatWidth:
                break
            if label == 'R':
                break
            if label == target_label:
                return pos

        return target_pos

    def GetNextLabelPosition(self, target_label, pos_list, label_list, r_pos):
        '''Get next target label from current r_pos.'''
        # pos_list must be sorted.
        r_index = bisect.bisect_left(pos_list, r_pos)

        # Find target_pos.
        target_pos = None
        for ind in xrange(r_index + 1, len(pos_list)):
            pos, label = (pos_list[ind], label_list[ind])
            if pos - r_pos > self.MaxHeartBeatWidth:
                break
            if label == 'R':
                break
            if label == target_label:
                return pos

        return target_pos

    def AddP2feature(
            self, rawsig, expert_marklist,
            debug = False,
            original_signal = None):
        '''Search backward from current R position and collect features.'''
        target_label = self.target_label
        N_expert_label = len(expert_marklist)

        # Get R position(index) list
        expert_labels = expert_marklist
        # expert labels must be sorted in acending order.
        pos_list, label_list = zip(*expert_labels)

        r_pos_list = map(lambda x:x[0],
                filter(lambda x: x[1] == 'R', expert_labels))

        # Reverse r_pos_list
        r_pos_list = r_pos_list[::-1]

        for r_pos in r_pos_list:
            # Get signal segment (segment between R-R).
            segment_range = self.GetRRSegmentPrev(r_pos_list, r_pos)
            target_pos = self.GetPreviousLabelPosition(
                    self.target_label,
                    pos_list, label_list, r_pos)

            # Skip if missing.
            if target_pos is None:
                continue

            # Extend segment
            segment_range[0] -= self.QRS_segment_expand_length
            segment_range[1] += self.QRS_segment_expand_length

            # Shift segment ranges to get more training samples
            base_segment_range = segment_range
            base_target_position = target_pos
            for shift_distance in xrange(0, 1, 1):
                segment_range[0] = base_segment_range[0] + shift_distance
                segment_range[1] = base_segment_range[1] + shift_distance

                # Target label position (Local)
                target_pos = base_target_position - segment_range[1] 

                # Get ECG raw signal segment
                sig_seg = self.CropSignalBySegmentRange(
                        rawsig,
                        segment_range,
                        self.FixedSignalLength,
                        limit_by_right = False)

                # Get cDlist segment.
                cDlist_seg = []
                for detail_signal in self.cDlist:
                    cDlist_seg.append(
                        self.CropSignalBySegmentRange(
                            detail_signal,
                            segment_range, self.FixedSignalLength,
                            limit_by_right = False))

                # Get QRS distance array.
                seg_len = segment_range[1] - segment_range[0] + 1

                QRS_distance_before = range(0, seg_len)
                QRS_distance_after = range(seg_len-1, -1, -1)
                QRS_distance_ratio = map(lambda x:float(x)/seg_len,QRS_distance_before)

                # Form current feature vector
                current_feature_vector = self.FormFeature(
                        cDlist_seg,
                        QRS_distance_before, QRS_distance_after, QRS_distance_ratio,
                        sig_seg,
                        modify_signal_tail = False,
                        local_target_pos = target_pos + seg_len,
                        debug = debug)
                if debug:
                    print 'rpos:', r_pos
                    print 'segment pair:', segment_range
                    # ECG
                    plt.figure(3)
                    plt.plot(rawsig)
                    max_val = np.max(rawsig)
                    min_val = np.min(rawsig)
                    plt.plot([segment_range[0],] * 2, [min_val, max_val], 'r--',
                            lw = 4)
                    plt.plot([segment_range[1],] * 2, [min_val, max_val], 'r--',
                            lw = 4)
                    plt.grid(True)
                    plt.title('ECG with segment')

                    fig1 = plt.figure(1)
                    fig1.clf()
                    subplot_total = len(cDlist_seg) + 1
                    for fig_i in xrange(0, subplot_total - 1):
                        plt.subplot(subplot_total, 1, fig_i + 1)
                        tmp_sig = cDlist_seg[fig_i]
                        plt.plot(tmp_sig)
                        plt.grid(True)
                    plt.subplot(subplot_total, 1, subplot_total)
                    plt.plot(sig_seg)
                    plt.title("Signal segment")
                    plt.show()
                    pdb.set_trace()

                # Training label postion
                current_output = target_pos
                # Add to Training Pool
                self.feature_pool_.append(current_feature_vector)
                self.output_pool_.append(current_output)
                # Logging
                log.info(
                        'Length of current training feature vector %d' % len(
                            current_feature_vector))
                log.info('Current output: %d' % current_output)
    def AddT2feature(
            self,
            rawsig,
            expert_marklist,
            debug = False,
            original_signal = None):
        # For each target_label:
        #  get QRS region before and after it.
        #  get segment signal& segment cDlist
        #  get QRS distance array
        target_label = self.target_label
        N_expert_label = len(expert_marklist)

        # Get R position(index) list
        expert_labels = expert_marklist
        # expert labels must be sorted in acending order.
        pos_list, label_list = zip(*expert_labels)

        r_labels = filter(lambda x: x[1] == 'R', expert_labels)
        r_pos_list = map(lambda x:x[0], r_labels)

        for r_pos in r_pos_list:
            # Get signal segment (segment between R-R).
            segment_range = self.GetRRSegment(r_pos_list, r_pos)
            target_pos = self.GetNextLabelPosition(
                    self.target_label,
                    pos_list,
                    label_list,
                    r_pos)
            # Skip if missing.
            if target_pos is None:
                continue
            # Extend segment
            segment_range[0] -= self.QRS_segment_expand_length
            segment_range[1] += self.QRS_segment_expand_length
            # Target label position
            target_pos -= segment_range[0] 

            # Get ECG raw signal segment
            sig_seg = self.CropSignalBySegmentRange(
                    rawsig,
                    segment_range,
                    self.FixedSignalLength)

            # Get cDlist segment.
            cDlist_seg = []
            for detail_signal in self.cDlist:
                cDlist_seg.append(
                    self.CropSignalBySegmentRange(
                        detail_signal,
                        segment_range,
                        self.FixedSignalLength))
            # Get QRS distance array.
            seg_len = segment_range[1] + 1 - segment_range[0]
            QRS_distance_before = range(0, seg_len)
            QRS_distance_after = range(seg_len-1, -1, -1)
            QRS_distance_ratio = map(lambda x:float(x)/seg_len,QRS_distance_before)
            # Form current feature vector
            current_feature_vector = self.FormFeature(
                    cDlist_seg,
                    QRS_distance_before,
                    QRS_distance_after,
                    QRS_distance_ratio,
                    sig_seg, debug = debug)
            # Training label postion
            current_output = target_pos
            # Add to Training Pool
            self.feature_pool_.append(current_feature_vector)
            self.output_pool_.append(current_output)
            # Logging
            log.info(
                    'Length of current training feature vector %d' % len(
                        current_feature_vector))
            log.info('Current output: %d' % current_output)

            if debug:
                print 'rpos:', r_pos
                print 'segment pair:', segment_range
                fig1 = plt.figure(1)
                fig1.clf()
                subplot_total = len(cDlist_seg) + 1
                for fig_i in xrange(0, subplot_total - 1):
                    plt.subplot(subplot_total, 1, fig_i + 1)
                    tmp_sig = cDlist_seg[fig_i]
                    plt.plot(tmp_sig)
                    plt.grid(True)
                plt.subplot(subplot_total, 1, subplot_total)
                plt.plot(sig_seg)
                plt.title("Signal segment")
                plt.show()
                pdb.set_trace()

            # debug
            if debug == True:
                # Check: 
                #   1. Position of T mark
                #   2. The segment of QRS
                plt.ion()
                plt.figure(1)
                plt.clf()

                # sub figure 1 (Removed QRS signal)
                plt.subplot(211)
                plt.title('segment [%d, %d]' % (segment_range[0], segment_range[1]))
                plt.plot(sig_seg,label = 'ECG')
                plt.plot(map(lambda x:x-segment_range[0],segment_range),
                        map(lambda x:sig_seg[x-segment_range[0]],segment_range),
                        'ro', label = 'R boundary')
                plt.plot(pos-segment_range[0],
                        sig_seg[pos-segment_range[0]],
                        'gd', label = self.target_label)
                plt.grid(True)
                # subplot 2 (orignal signal)
                plt.subplot(212)
                plt.plot(original_signal[segment_range[0]:segment_range[1]+1])
                plt.grid(True)

                plt.legend()
                plt.show()
                pdb.set_trace()

                    


    # Note That boundary_function is either max or min
    # pos is current start position.
    def FillInBlankQRSMark(self, pos, qrs_label_dict, boundary_function):
        '''
        Fill the missing QRS mark(s).

        Example:
        If next Q & S exists but R missing:
        next QRS: --> boundary_function(Q_pos, S_pos)
        next QRS: -->               min(Q_pos, S_pos)
        next QRS: -->               max(Q_pos, S_pos)
        '''
        # Logging
        log.debug('--')
        log.debug('Original qrs_label_dict:')
        log.debug(qrs_label_dict)

        qrs_single_position = None
        for qrs_label in self.QRSlabels:
            if qrs_label in qrs_label_dict:
                if qrs_single_position is None:
                    qrs_single_position = qrs_label_dict[qrs_label]
                else:
                    qrs_single_position = boundary_function(qrs_single_position,
                            qrs_label_dict[qrs_label])
        for qrs_label in self.QRSlabels:
            if qrs_label not in qrs_label_dict:
                if qrs_single_position is None:
                    qrs_single_position = pos + self.MaxHeartBeatWidth/2
                qrs_label_dict[qrs_label] = qrs_single_position
                log.warning('Adding {} to {}'.format(qrs_label,qrs_single_position))
        return qrs_label_dict
    def CropFeatureWindow(self,
            cDlist_seg,
            QRS_distance_before,
            QRS_distance_after,
            QRS_distance_ratio,
            sig_seg,
            modify_signal_tail = True):
        '''Make sure the input signals have equal length of self.FixedSignalLength.'''

        def CropEcgWaveForm(signal_segment, fixed_length):
            '''Crop gc function for ECG waveform, detail signal.'''
            N_current = len(signal_segment)
            if N_current < fixed_length:
                # Stuffing polyfoam behind.
                # log.warning('Heart Beat Segment is too short, stuffing constants behind!')
                signal_segment.extend([signal_segment[-1],] * (fixed_length - N_current))
            elif N_current > fixed_length:
                # Cutoff from behind
                # log.warning('Heart Beat Segment is too long, cutting off the tail part!')
                signal_segment = signal_segment[0:fixed_length]
            return signal_segment

        def CropEcgWaveFormByPrev(signal_segment, fixed_length):
            '''Crop gc function for ECG waveform, detail signal.'''
            N_current = len(signal_segment)
            if N_current < fixed_length:
                # Stuffing polyfoam behind.
                # log.warning('Heart Beat Segment is too short, stuffing constants behind!')
                padding_list = [signal_segment[0],] * (fixed_length - N_current)
                padding_list.extend(signal_segment)
                signal_segment = padding_list
            elif N_current > fixed_length:
                # Cutoff from behind
                # log.warning('Heart Beat Segment is too long, cutting off the tail part!')
                signal_segment = signal_segment[-fixed_length:len(signal_segment)]

            # debug
            # print 'fixed_length:', fixed_length
            # print 'type signal_segment:', type(signal_segment)
            # print 'len signal_segment:', len(signal_segment)
            # pdb.set_trace()

            return signal_segment

        def NormalizeAmplitude(signal_segment, amplitude_max, amplitude_min):
            '''Normalize signal amplitude.'''
            # amplitude_max = max(signal_segment)
            # amplitude_min = min(signal_segment)
            if abs(amplitude_max - amplitude_min) < 1e-6:
                # Amplitude difference is too small!
                log.warning('Amplitude difference of signal segment is too small!')
                return signal_segment
            signal_segment = [(val - amplitude_min)/(amplitude_max - amplitude_min)
                    for val in signal_segment]
            return signal_segment
            
        fixed_length = self.FixedSignalLength
        if modify_signal_tail:
            sig_seg = CropEcgWaveForm(sig_seg, fixed_length)
        else:
            sig_seg = CropEcgWaveFormByPrev(sig_seg, fixed_length)

        amplitude_max = max(sig_seg)
        amplitude_min = min(sig_seg)
        sig_seg = NormalizeAmplitude(sig_seg, amplitude_max, amplitude_min)

        modified_swt_detail_list = []
        for detail_signal_segment in cDlist_seg:
            # Modifies the detail_signal_segment.
            if modify_signal_tail:
                detail_signal_segment = CropEcgWaveForm(
                        detail_signal_segment,
                        fixed_length)
            else:
                detail_signal_segment = CropEcgWaveFormByPrev(
                        detail_signal_segment,
                        fixed_length)
            detail_signal_segment = NormalizeAmplitude(
                    detail_signal_segment,
                    amplitude_max, amplitude_min)
            modified_swt_detail_list.append(detail_signal_segment)
            
        return (modified_swt_detail_list,
                QRS_distance_before, QRS_distance_after, QRS_distance_ratio,
                sig_seg)
            
    def FormFeature(
                self, cDlist_seg,
                QRS_distance_before, QRS_distance_after, QRS_distance_ratio,
                sig_seg,
                modify_signal_tail = True,
                local_target_pos = None,
                debug = False):
        '''Form feature from the segment signal given.
            Inputs:
                local_target_pos: The index of local target index, in range of
                sig_seg.
        '''
        # Normalize the length of each input feature signal.
        (
            cDlist_seg,
            QRS_distance_before, QRS_distance_after, QRS_distance_ratio,
            sig_seg
        ) = self.CropFeatureWindow(
                  cDlist_seg,
                  QRS_distance_before, QRS_distance_after, QRS_distance_ratio,
                  sig_seg,
                  modify_signal_tail = modify_signal_tail)

        # Logging
        # log.info('Only extracting features from SWT level 2 to 6.')
        feature_vector = []

        conf = dict()
        conf['regression_swt_feature'] = 'difference'
        if conf['regression_swt_feature'] == 'difference':
            for detail_level in cDlist_seg[1:6]:
                detail_level = np.array(detail_level)
                # Add Pairs fixed right position.
                for right_pos in xrange(-7, -17, -2):
                    diff_vec = detail_level - detail_level[right_pos]
                    abs_vec = np.abs(diff_vec)

                    feature_vector.extend(diff_vec[::2])
                    feature_vector.extend(abs_vec[::2])

        elif conf['regression_swt_feature'] == 'amplitude':
            for detail_level in cDlist_seg[1:6]:
                feature_vector.extend(detail_level)
                feature_vector.extend(np.abs(detail_level))
        # Add distance to QRS feature.
        # feature_vector.extend(QRS_distance_after)
        # feature_vector.extend(QRS_distance_ratio)
        # feature_vector.extend(QRS_distance_before)
        feature_vector.extend(sig_seg)
        # debug
        if debug == True:
            plt.figure(2)
            plt.clf()
            N_subplot_level = 6
            plt.subplot(N_subplot_level, 1, 1)
            plt.plot(sig_seg)
            plt.grid(True)

            for di in range(1, 6):
                plt.subplot(N_subplot_level, 1, di + 1)
                plt.plot(cDlist_seg[di])
                plt.grid(True)

        # Read bias distibution from file
        # Add to feature_vector(Assume this classifier is for P wave)
        if self.target_label != 'P':
            pass
            # raise Exception("classifier target label is not P!")
        else :
            # Add Gaussian probability vector based on local_target_pos
            g_mean = float(local_target_pos)
            g_std = 20.0
            g_prob = list()
            for ind in xrange(0, len(sig_seg)):
                val = 1.0 / math.sqrt(2 * math.pi) / g_std
                val *= math.exp( - (ind - g_mean) ** 2 / 2 / g_std ** 2)
                g_prob.append(val)
        
            feature_vector.extend(g_prob)

        if local_target_pos is None:
            feature_vector.extend([0,] * len(sig_seg))
            return feature_vector

        return feature_vector

    def check_validQRS(self, label_dict):
        '''Check QRS in label_dict is valid.'''
        if all(map(lambda x:x in label_dict, self.QRSlabels)) == False:
            print label_dict
            raise Exception('QRS not all in label_dict!')
        # if label_dict['Ronset'] > label_dict['R'] or label_dict['R'] - label_dict['Ronset'] > self.MaxQRSWidth:
            # print label_dict
            # raise Exception('QRS mispositioned!')
        # if label_dict['R'] > label_dict['Roffset'] or label_dict['Roffset'] - label_dict['R'] > self.MaxQRSWidth:
            # print label_dict
            # raise Exception('QRS mispositioned!')
            
                


    def training(self):
        '''Train RandomForestClassifier.'''
        self.rfclassifier = RandomForestRegressor(
                n_estimators = self.RF_TreeNumber,
                max_depth = self.Tree_Max_Depth,
                n_jobs =4,
                warm_start = False)
        # Logging
        log.info('Feature matrix size: [%d * %d]' % (
            len(self.feature_pool_),
            len(self.feature_pool_[0])))
        self.rfclassifier.fit(self.feature_pool_,self.output_pool_)
        self.mdl = self.rfclassifier

    def testing_P_backward(self, rawsig, expert_marklist):
        '''Testing by regression forward: used for testing T waves.'''
        expert_marklist.sort(key = lambda x:x[0])
        # Set self.cDlist
        self.nonQRS_swt(rawsig,expert_marklist)
        # Output
        output_prediction_list = []
        # get QRS region.
        N_expert_label = len(expert_marklist)

        # Get R position(index) list
        expert_labels = expert_marklist
        # expert labels must be sorted in acending order.
        pos_list, label_list = zip(*expert_labels)

        r_labels = filter(lambda x: x[1] == 'R', expert_labels)
        r_pos_list = map(lambda x:x[0], r_labels)

        # Benchmark information
        number_of_beats_tested = len(r_pos_list)
        testing_time_cost = time.time()
        
        # Collect sample features into buckets
        feature_buckets = []
        result_bias_list = []

        for r_pos in r_pos_list[::-1]:

            # Get signal segment.
            segment_range = self.GetRRSegmentPrev(r_pos_list, r_pos)
            # Segment is from Q_before to S_after.
            segment_range[0] -= self.QRS_segment_expand_length
            segment_range[1] += self.QRS_segment_expand_length

            # Get raw ECG signal segment
            sig_seg = self.CropSignalBySegmentRange(
                    rawsig,
                    segment_range,
                    self.FixedSignalLength,
                    limit_by_right = False)

            # Get cDlist segment.
            cDlist_seg = []
            for detail_signal in self.cDlist:
                cDlist_seg.append(
                    self.CropSignalBySegmentRange(
                        detail_signal,
                        segment_range,
                        self.FixedSignalLength,
                        limit_by_right = False))
            # Get QRS distance array.
            seg_len = segment_range[1] + 1 - segment_range[0]
            QRS_distance_before = range(0, seg_len)
            QRS_distance_after = range(seg_len-1, -1, -1)
            QRS_distance_ratio = map(lambda x:float(x)/seg_len,QRS_distance_before)
            # Form current feature vector[P]
            current_feature_vector = self.FormFeature(
                    cDlist_seg,
                    QRS_distance_before,
                    QRS_distance_after,
                    QRS_distance_ratio,
                    sig_seg,
                    local_target_pos = seg_len - 44,
                    modify_signal_tail = False)


            # Collect samples
            feature_buckets.append(current_feature_vector)
            result_bias_list.append(segment_range[1])


        # Filter empty samples
        feature_buckets = filter(lambda x: len(x) > 0, feature_buckets)
        # Bucket testing
        if len(feature_buckets) > 0:
            predict_results = self.rfclassifier.predict(feature_buckets)
            predict_results += np.array(result_bias_list)
            # Return result as Python list
            output_prediction_list = predict_results.tolist()
        

        # Output time cost info to log
        testing_time_cost = time.time() - testing_time_cost

        if testing_time_cost == 0:
            # Avoid division by 0
            testing_time_cost = 1
        average_time_cost = float(number_of_beats_tested) / float(testing_time_cost)
        log.info('Testing speed: %f beats per second.' % average_time_cost)

        return output_prediction_list
    def LoadModel(self, model_object):
        '''Read a pre-trained random forest model.'''
        self.mdl = self.rfclassifier = model_object
    def testing_T_forward(self, rawsig, expert_marklist):
        '''Testing by regression forward: used for testing T waves.'''
        expert_marklist.sort(key = lambda x:x[0])
        # Set self.cDlist
        self.nonQRS_swt(rawsig,expert_marklist)
        # Output
        output_prediction_list = []
        # get QRS region.
        N_expert_label = len(expert_marklist)

        # Get R position(index) list
        expert_labels = expert_marklist
        # expert labels must be sorted in acending order.
        pos_list, label_list = zip(*expert_labels)

        r_labels = filter(lambda x: x[1] == 'R', expert_labels)
        r_pos_list = map(lambda x:x[0], r_labels)

        # Benchmark information
        number_of_beats_tested = len(r_pos_list)
        testing_time_cost = time.time()
        
        # Collect sample features into buckets
        feature_buckets = []
        result_bias_list = []
        for r_pos in r_pos_list:

            # Get signal segment.
            segment_range = self.GetRRSegment(r_pos_list, r_pos)
            # Segment is from Q_before to S_after.
            segment_range[0] -= self.QRS_segment_expand_length
            segment_range[1] += self.QRS_segment_expand_length

            # Get raw ECG signal segment
            sig_seg = self.CropSignalBySegmentRange(
                    rawsig,
                    segment_range,
                    self.FixedSignalLength)

            # Get cDlist segment.
            cDlist_seg = []
            for detail_signal in self.cDlist:
                cDlist_seg.append(
                    self.CropSignalBySegmentRange(
                        detail_signal,
                        segment_range,
                        self.FixedSignalLength))
            # Get QRS distance array.
            seg_len = segment_range[1] + 1 - segment_range[0]
            QRS_distance_before = range(0, seg_len)
            QRS_distance_after = range(seg_len-1, -1, -1)
            QRS_distance_ratio = map(lambda x:float(x)/seg_len,QRS_distance_before)
            # Form current feature vector
            current_feature_vector = self.FormFeature(
                    cDlist_seg,
                    QRS_distance_before,
                    QRS_distance_after,
                    QRS_distance_ratio,
                    sig_seg)


            # Collect samples
            feature_buckets.append(current_feature_vector)
            result_bias_list.append(segment_range[0])


        # Filter empty samples
        feature_buckets = filter(lambda x: len(x) > 0, feature_buckets)
        # Bucket testing
        if len(feature_buckets) > 0:
            predict_results = self.rfclassifier.predict(feature_buckets)
            predict_results += np.array(result_bias_list)
            # Return result as Python list
            output_prediction_list = predict_results.tolist()
        

        # Output time cost info to log
        testing_time_cost = time.time() - testing_time_cost

        if testing_time_cost == 0:
            # Avoid division by 0
            testing_time_cost = 1
        average_time_cost = float(number_of_beats_tested) / float(testing_time_cost)
        log.info('Testing speed: %f beats per second.' % average_time_cost)

        return output_prediction_list

    def testing(self, rawsig, expert_marklist):
        '''Testing ECG.'''
        if self.target_label in ['T', 'Toffset', 'Tonset']:
            return self.testing_T_forward(rawsig, expert_marklist)
        if self.target_label in ['P', 'Poffset', 'Ponset']:
            return self.testing_P_backward(rawsig, expert_marklist)
        raise Exception('Unknown target_label (%s)' % self.target_label)

    def FlatternAroundR(self, rawsig, MarkList, QRS_width_threshold = 180):
        '''Marklist only contains R labels.'''
        expert_marklist = MarkList
        QRS_fixed_length = 20

        # Use QRS region to flattern the signal
        expert_marklist = filter(lambda x:'R' == x[1], expert_marklist)
        # Get QRS region
        expert_marklist.sort(key=lambda x:x[0])
        len_sig = len(rawsig)
        N_Rlist = len(expert_marklist)

        for ind in xrange(0,N_Rlist):
            current_pos = expert_marklist[ind][0]
            region_left = max(current_pos - QRS_fixed_length, 0)
            region_right = min(current_pos + QRS_fixed_length, len_sig - 1)

            if region_left > region_right:
                continue
            # 
            pos, next_pos = region_left, region_right
            amp_start = rawsig[pos]
            amp_end = rawsig[next_pos]
            flat_segment = map(
                    lambda x:amp_start+float(x)*(amp_end-amp_start)/(next_pos-pos),
                    xrange(0,next_pos-pos))
            for segment_index in xrange(pos,next_pos):
                rawsig[segment_index] = flat_segment[segment_index-pos]


        return rawsig

    def getNonQRSsig(self,rawsig, MarkList, QRS_width_threshold = 180):
        # QRS width threshold:
        #   The max possible distance between Q & S,
        #    otherwise there's no matching S for Q.

        expert_marklist = MarkList
        
        # Use QRS region to flattern the signal, find 'Ronset' & 'Roffset'
        expert_marklist = filter(lambda x:'R' in x[1] and len(x[1])>1,expert_marklist)
        # Get QRS region
        expert_marklist.sort(key=lambda x:x[0])
        QRS_regionlist = []
        N_Rlist = len(expert_marklist)
        for ind in xrange(0,N_Rlist-1):
            pos, label = expert_marklist[ind]
            # last one: no match pair
            if ind == N_Rlist - 1:
                break
            elif label != 'Ronset':
                continue
            # get next label
            next_pos,next_label = expert_marklist[ind+1]
            if next_label == 'Roffset':
                if next_pos - pos >=QRS_width_threshold:
                    log.warning('Warning: no matching Roffset found!')
                else:
                    QRS_regionlist.append([pos,next_pos])
                    # print 'Adding:',pos,next_pos
                    # flattern the signal
                    amp_start = rawsig[pos]
                    amp_end = rawsig[next_pos]
                    flat_segment = map(
                            lambda x:amp_start+float(x)*(amp_end-amp_start)/(next_pos-pos),
                            xrange(0,next_pos-pos))
                    for segment_index in xrange(pos,next_pos):
                        rawsig[segment_index] = flat_segment[segment_index-pos]
        return rawsig

    def crop_data_for_swt(self,rawsig):
        # crop rawsig, make the len(rawsig) is 2^N.
        base2 = 1
        N_data = len(rawsig)
        if len(rawsig)<=1:
            raise Exception('len(rawsig)={}, too short for swt!',len(rawsig))
        crop_len = base2
        while base2<N_data:
            if base2*2>=N_data:
                crop_len = base2*2
                break
            base2*=2
        # pad zeros
        if N_data< crop_len:
            rawsig += [rawsig[-1],]*(crop_len-N_data)
        return rawsig

    def nonQRS_swt(self,rawsig,expert_marklist,wavelet = 'db6',MaxLevel = 9):
        '''Get swt without QRS regions, modify rawsig whthin this function.'''
        # Get Swt coef for rawsig.
        # rawsig = self.getNonQRSsig(rawsig,expert_marklist)
        rawsig = self.FlatternAroundR(rawsig,expert_marklist)
        # plt.plot(rawsig)
        # plt.show()

        rawsig = self.crop_data_for_swt(rawsig)

        coeflist = pywt.swt(rawsig,wavelet,MaxLevel)
        cAlist,cDlist = zip(*coeflist)
        # append to self.cDlist
        self.cAlist = []
        self.cDlist = []
        for ind in xrange(0,len(cAlist)):
            self.cAlist.append(cAlist[ind].tolist())
            self.cDlist.append(cDlist[ind].tolist())

        self.cDlist = self.cDlist[::-1]
        self.cAlist = self.cAlist[::-1]
        return None

if __name__ == '__main__':
    pass
