# -*- coding:utf-8 -*-  
import os
import sys
import codecs

import matplotlib
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import pdb
import math
import glob
import json


zn_font = matplotlib.font_manager.FontProperties(
        fname = '/usr/share/fonts/truetype/simsun.ttc')

class QrsTypeDetector(object):
    '''Detection of characteristic points in QRS complex.'''
    def __init__(self, fs):
        '''Temp init.'''
        self.fs = fs
    # def __init__(self, raw_sig, r_pos_list, record_ID, fs):
        # '''Diagnosis Common Diseases.'''
        # self.record_ID = record_ID
        # self.fs = fs

        # self.result_dict = dict()
        # self.result_dict['R'] = r_pos_list

        # self.diagnosis_text = ""


    def write_progress(self):
        '''Write diagnosis progress to file.'''
        with open('/tmp/diag_progress.json', 'w') as fout:
            json.dump(self.diag_progress, fout)
        
        
    def SearchForQ(self, raw_sig, q_index, r_index, 
            searched_r,
            MaxSearchLengthMs = 10):
        # Search for q before current r
        MaxSearchIndexLen = MaxSearchLengthMs / 1000.0 * self.fs
        left_bound = max(1, q_index - 10)
        q_position_before_r = None
        EliminationOpp = 5
        q_search_left_length = 10
        q_search_leftpeak_length = 5
        FlatThreshold = 5

        for ind in xrange(searched_r - 1, left_bound - 1, -1):
            # Downward peak
            is_peak = False
            if raw_sig[ind] <= raw_sig[ind - 1] and raw_sig[ind] <= raw_sig[ind + 1]:
                is_peak = True
            is_flat = False
            flat_left = max(0, ind - q_search_left_length)
            if (len(raw_sig[flat_left:ind]) == 0 or
                    raw_sig[ind] - np.min(raw_sig[flat_left:ind]) <= FlatThreshold):
                is_flat = True
                
            if is_peak == False:
                if is_flat == False:
                    continue
                else:
                    # Search for nearby peak
                    have_nearby_peak = False
                    for ppeak_ind in xrange(ind - 1, max(1, ind - q_search_leftpeak_length), -1):
                        if (raw_sig[ppeak_ind] <= raw_sig[ppeak_ind - 1] and
                                raw_sig[ppeak_ind] <= raw_sig[ppeak_ind + 1]):
                            have_nearby_peak = True
                            break
                    if have_nearby_peak == False:
                        # print 'Is Flat!'
                        q_position_before_r = ind
                        break
                    else:
                        continue
            if is_flat:
                # print 'Is Flat!'
                q_position_before_r = ind
                break

            # Not flat peak
            if EliminationOpp <= 0:
                # print 'Elimination Exausted!'
                q_position_before_r = ind
                break
            flat_left = max(1, ind - q_search_leftpeak_length)
            left_positive_peak_ind = None
            for ppeak_ind in xrange(ind - 1, flat_left - 1, -1):
                if (raw_sig[ppeak_ind] >= raw_sig[ppeak_ind - 1] and
                        raw_sig[ppeak_ind] >= raw_sig[ppeak_ind + 1]):
                    left_positive_peak_ind = ppeak_ind
                    break
            if left_positive_peak_ind is None:
                # print 'No continous left peak!'
                q_position_before_r = ind
                break
            EliminationOpp -= 1

        if q_position_before_r is None:
            # q
            left_bound = max(0, q_index - MaxSearchIndexLen)
            right_bound = min(r_index, q_index + MaxSearchIndexLen)
            left_bound = int(left_bound)
            right_bound = int(right_bound)
            searched_q = np.argmin(raw_sig[left_bound:right_bound + 1]) + left_bound
            return searched_q
        else:
            return q_position_before_r

    def SearchForS(self, raw_sig, s_index, r_index, 
            searched_r,
            MaxSearchLengthMs = 10):
        '''Search for S after current r.'''
        MaxSearchIndexLen = MaxSearchLengthMs / 1000.0 * self.fs
        # left_bound = max(1, q_index - 10)
        len_sig = len(raw_sig)
        right_bound = min(len_sig - 2, s_index + 10)
        s_position_after_r = None
        EliminationOpp = 5
        s_search_length = 10
        s_search_peak_length = 5
        FlatThreshold = 5

        for ind in xrange(searched_r + 1, right_bound + 1):
            # Downward peak
            is_peak = False
            if raw_sig[ind] <= raw_sig[ind - 1] and raw_sig[ind] <= raw_sig[ind + 1]:
                is_peak = True
            is_flat = False
            flat_right = min(len_sig - 2, ind + s_search_length)
            if (len(raw_sig[ind + 1:flat_right + 1]) == 0 or
                    raw_sig[ind] - np.min(raw_sig[ind + 1:flat_right + 1]) <= FlatThreshold):
                is_flat = True
                
            if is_peak == False:
                if is_flat == False:
                    continue
                else:
                    # Search for nearby peak
                    have_nearby_peak = False
                    for ppeak_ind in xrange(ind + 1, min(len_sig - 2, ind + s_search_peak_length)):
                        if (raw_sig[ppeak_ind] <= raw_sig[ppeak_ind - 1] and
                                raw_sig[ppeak_ind] <= raw_sig[ppeak_ind + 1]):
                            have_nearby_peak = True
                            break
                    if have_nearby_peak == False:
                        # print 'Is Flat!'
                        s_position_after_r = ind
                        break
                    else:
                        continue
            if is_flat:
                # print 'Is Flat!'
                s_position_after_r = ind
                break

            # Not flat peak
            if EliminationOpp <= 0:
                # print 'Elimination Exausted!'
                s_position_after_r = ind
                break
            flat_right = min(len_sig - 2, ind + s_search_peak_length)
            right_positive_peak_ind = None
            for ppeak_ind in xrange(ind + 1, flat_right + 1):
                if (raw_sig[ppeak_ind] >= raw_sig[ppeak_ind - 1] and
                        raw_sig[ppeak_ind] >= raw_sig[ppeak_ind + 1]):
                    right_positive_peak_ind = ppeak_ind
                    break
            if right_positive_peak_ind is None:
                # print 'No continous right peak!'
                s_position_after_r = ind
                break
            EliminationOpp -= 1

        if s_position_after_r is None:
            # s
            left_bound = searched_r
            right_bound = min(len_sig - 1, s_index + MaxSearchIndexLen)
            left_bound = int(left_bound)
            right_bound = int(right_bound)
            searched_s = np.argmin(raw_sig[left_bound:right_bound + 1]) + left_bound
            return searched_s
        else:
            return s_position_after_r

    def GetQrsType(self, raw_sig, q_index, r_index, s_index,
            MaxAmplitude = 400, MaxSearchLengthMs = 10, debug_plot = False):
        '''Get type of QRS such as rsR'.'''
        raw_sig = np.array(raw_sig) * 1000.0
        r_index = int(r_index)
        q_index = int(q_index)
        s_index = int(s_index)

        MaxAmplitude = np.nanmax(raw_sig)
        Q_amplitude_threshold = 100.0
        LargeWaveThreshold = 250.0

        MaxSearchIndexLen = MaxSearchLengthMs / 1000.0 * self.fs
        R_MaxSearchIndexLen = 2 * MaxSearchIndexLen
        new_pos_list = []
        qrs_type = ''

        # q
        left_bound = max(0, q_index - MaxSearchIndexLen)
        right_bound = min(r_index, q_index + MaxSearchIndexLen)
        left_bound = int(left_bound)
        right_bound = int(right_bound)
        searched_q = np.argmin(raw_sig[left_bound:right_bound + 1]) + left_bound
        # default value, update it later
        qrs_type += ' ';
        new_pos_list.append(searched_q)
            
        # r
        left_bound = q_index
        # left_bound = max(q_index, r_index - R_MaxSearchIndexLen)
        right_bound = min(s_index, r_index + MaxSearchIndexLen)
        left_bound = int(left_bound)
        right_bound = int(right_bound)
        searched_r = np.argmax(raw_sig[left_bound:right_bound + 1]) + left_bound
        r_amplitude = raw_sig[searched_r]
        if abs(r_amplitude) > LargeWaveThreshold:
            qrs_type += 'R'
        else:
            qrs_type += 'r'
        new_pos_list.append(searched_r)

        q_position_before_r = self.SearchForQ(raw_sig, q_index, r_index, searched_r)
            
        old_q = searched_q
        if q_position_before_r is not None:
            # print 'Updated q!'
            searched_q = q_position_before_r
            new_pos_list[0] = q_position_before_r

        # Update Q type
        q_search_left_length = 5
        MinValleyDepth = 28
        q_type = ' '
        q_amplitude = raw_sig[searched_q]
        left_bound = max(0, searched_q - q_search_left_length)
        right_bound = min(len(raw_sig) - 1, searched_q + q_search_left_length)

        if (len(raw_sig[left_bound:searched_q]) == 0 or
                np.max(raw_sig[left_bound:searched_q]) - q_amplitude < MinValleyDepth):
            q_type = ' '
        elif (len(raw_sig[searched_q + 1:right_bound]) == 0 or
                np.max(raw_sig[searched_q + 1:right_bound]) - q_amplitude < MinValleyDepth):
            q_type = ' '
        else:
            if abs(q_amplitude) > LargeWaveThreshold:
                q_type = 'Q'
            else:
                q_type = 'q'
        qrs_type = q_type + qrs_type[1:]


        
        # s
        # left_bound = max(r_index, s_index - MaxSearchIndexLen)
        left_bound = searched_r
        right_bound = min(len(raw_sig) - 1, s_index + MaxSearchIndexLen)
        left_bound = int(left_bound)
        right_bound = int(right_bound)
        # searched_s = np.argmin(raw_sig[left_bound:right_bound + 1]) + left_bound
        # searched_s = self.
        searched_s = self.SearchForS(raw_sig, s_index, r_index, searched_r)

        s_amplitude = raw_sig[searched_s]
        new_pos_list.append(searched_s)
        # Update S type
        search_left_length = 5
        MinValleyDepth = 28
        s_type = ' '
        s_amplitude = raw_sig[searched_s]
        left_bound = max(0, searched_s - search_left_length)
        right_bound = min(len(raw_sig) - 1, searched_s + search_left_length)
        if s_amplitude < 220:
            if abs(s_amplitude) > LargeWaveThreshold:
                s_type = 'S'
            else:
                s_type = 's'
        elif (len(raw_sig[left_bound:searched_s]) == 0 or
                np.max(raw_sig[left_bound:searched_s]) - s_amplitude < MinValleyDepth):
            s_type = ' '
        elif (len(raw_sig[searched_s + 1:right_bound]) == 0 or
                np.max(raw_sig[searched_s + 1:right_bound]) - s_amplitude < MinValleyDepth):
            s_type = ' '
        else:
            if abs(s_amplitude) > LargeWaveThreshold:
                s_type = 'S'
            else:
                s_type = 's'
        qrs_type += s_type
        
        if debug_plot == True:
            # debug
            print 'QRS type :', qrs_type
            plt.figure(3)
            plt.clf()
            plt.plot(raw_sig)
            labels = ['Ronset', 'R', 'Roffset']
            for xpos, label in zip(new_pos_list, labels):
                plt.plot(xpos, raw_sig[xpos], 'o', markersize = 12, label = label)

            plt.plot(old_q, raw_sig[old_q], 'x', markersize = 17, label = 'Old Q')
            plt.plot(s_index, raw_sig[s_index], 'x', markersize = 17, label = 'S_index')
            # Annotate qrs types
            for searched_pos, current_label in zip(new_pos_list, qrs_type):
                text_height = 100
                if current_label not in ['r', 'R']:
                    text_height = -100
                if current_label == ' ':
                    continue
                plt.annotate(current_label, xy=(searched_pos, raw_sig[searched_pos]),
                        xytext=(searched_pos-7, raw_sig[searched_pos] + text_height),
                        arrowprops=dict(facecolor='red', shrink=0.05, width = 3,
                            alpha = 0.5, linewidth = 0),
                        fontsize = 19,
                        bbox = dict(edgecolor = 'black', facecolor = 'none'),
                        )
            plt.legend()
            plt.xlim((min(new_pos_list) - 30, max(new_pos_list) + 30))
            plt.show(block = False)

        return ((searched_q, searched_r, searched_s), qrs_type.strip())
        
    def GetQrsDegree(self, q_index, r_index, s_index, MaxSearchLengthMs = 10):
        '''Get Degree.'''

        # Get sum of QRS in II
        pos_list = (q_index, r_index, s_index)
        raw_sig = self.sig_data['aVF']

        if self.record_ID not in self.diag_progress:
            self.diag_progress[self.record_ID] = dict()
        if 'AVF' not in self.diag_progress[self.record_ID]:
             self.diag_progress[self.record_ID]['AVF'] = list()
        if r_index not in self.diag_progress[self.record_ID]['AVF']:
            new_pos_list, qrs_type = self.GetQrsType(self.sig_data['aVF'],
                    q_index, r_index, s_index,
                    MaxSearchLengthMs = MaxSearchLengthMs, debug_plot = False)
            sum_aVF = sum([raw_sig[x] for x in new_pos_list])
        
            self.diag_progress[self.record_ID]['AVF'].append(r_index)
            # self.write_progress()

            # Get sum of QRS in I
            raw_sig = self.sig_data['I']
            new_pos_list, qrs_type = self.GetQrsType(raw_sig,
                    q_index, r_index, s_index,
                    MaxSearchLengthMs = MaxSearchLengthMs)
            sumI = sum([raw_sig[x] for x in new_pos_list])

            if abs(sumI) < 1e-6:
                if abs(sum_aVF) < 1e-6:
                    # Uncertain
                    return None
                elif sum_aVF < 0:
                    theta = -math.pi / 2.0
                else:
                    theta = math.pi / 2.0
            else:
                # value of tan_theta
                atan_x = 2.0 / math.sqrt(3.0) * (float(sum_aVF) / sumI)
                # atan_x = float(sum_aVF) / sumI
            
                if (sumI > 0 and sum_aVF > 0):
                    theta = math.atan(abs(atan_x))
                elif (sumI > 0 and sum_aVF < 0):
                    theta = -math.atan(abs(atan_x))
                elif (sumI < 0 and sum_aVF > 0):
                    theta = math.pi - math.atan(abs(atan_x))
                else:
                    theta = math.pi + math.atan(abs(atan_x))


            theta = theta / math.pi * 180.0
            # print 'theta = ', theta

            # amplist = [raw_sig[x] for x in pos_list]
            # plt.plot(raw_sig)
            # plt.plot(pos_list, amplist, 'ro', markersize = 12)
            # plt.show()
            # pdb.set_trace()
            return theta
        return None

    def Diag_DegreeList(self, MaxQRSGapMs = 500, SearchLengthMs = 10):
        '''Get Diagnosis Degree.'''
        pre_ronset_pos = -1
        pre_r_pos = -1
        len_results = len(self.results)
        
        degree_list = list()

        for ind in xrange(0, len_results):
            pos, label, xx = self.results[ind]
            if label == 'Ronset':
                pre_ronset_pos = pos
                pre_r_pos = -1
            elif label == 'R':
                pre_r_pos = pos
            elif label == 'Roffset':
                if (pre_ronset_pos != -1 and
                        pre_r_pos != -1 and
                        pos - pre_ronset_pos < MaxQRSGapMs / 1000.0 * self.fs):
                    # Is a Ronset-Roffset pair
                    # Search for actual positions of (q, r, s)
                    current_degree = self.GetQrsDegree(pre_ronset_pos, pre_r_pos, pos)
                    if current_degree is not None:
                        degree_list.append(current_degree)
        return degree_list
                    
                    

    def Diag_Arrithmia(self):
        '''Diagnosis.'''
        r_poslist = self.result_dict['R']
        if len(r_poslist) < 2:
            raise Exception('Not enough R wave detected!')

        RR_period_list = [x[1] - x[0] for x in zip(r_poslist, r_poslist[1:])]
        
        diag_diff = max(RR_period_list) - min(RR_period_list)

        if diag_diff / self.fs * 1000.0 >= 120:
            return True
        else:
            return False

        



def ParseDegree(diagnosis_text):
    '''Parse the degree number value.'''
    import re
    pat = re.compile(r'([\-+0-9]+)')
    results = pat.search(diagnosis_text)

    return int(results.group(0))
    
def ShowSignal(folder_path, record_ID):
    '''Plot signal.'''
    mat_file = glob.glob(os.path.join(folder_path, record_ID + '*.mat'))
    if len(mat_file) == 0:
        print 'No mat file matching current ID:', record_ID
        return
    import scipy.io as sio
    sig = sio.loadmat(mat_file[0])
    raw_sig = np.squeeze(sig['II'])
    plt.plot(raw_sig)
    plt.show()
    
    
    
def Test():
    '''Test funcion for QrsTypeDetector.'''
    with codecs.open('./diagnosis_info.json', 'r', 'utf8') as fin:
        dinfo = json.load(fin)

    # debug
    debug_count = 500000
    TP_count = 0
    FN_count = 0
    FP_count = 0
    total_count = 0


    # debug
    degree_diffs = list()

    for diagnosis_text, file_path in dinfo:
        if diagnosis_text is not None:
            file_short_name = os.path.split(file_path)[-1]
            current_folder = os.path.split(file_path)[0]
            mat_file_name = file_short_name.split('.')[0]
            if '_' in mat_file_name:
                mat_file_name = mat_file_name.split('_')[0]
            record_ID = mat_file_name

            # Load results
            result_file_path = os.path.join(current_folder, record_ID + '_results.json')
            # Result file not exist
            if os.path.exists(result_file_path) == False:
                continue

            diag = QrsTypeDetector(result_file_path, record_ID)
            if u'电轴' not in diagnosis_text or record_ID.startswith('TJ'):
                continue

            diagnosis_heart_degree = ParseDegree(diagnosis_text)
            # Skip negtive degrees
            # if '-' in diagnosis_text:
                # continue

            # print 'diagnosis:', diagnosis_text
            # print file_path
            if True or debug_count <= 14e5:
                degrees = diag.Diag_DegreeList()
                print degrees
                mean_degree = np.nanmean(degrees)
                degree_diffs.append(mean_degree - diagnosis_heart_degree)

                print '\n'
                print 'Mean degree:', np.nanmean(degrees)
                print '[%d]' % (mean_degree - diagnosis_heart_degree)
                print '\n'
                # pdb.set_trace()

            # Statistics
            total_count += 1

                
            # print '@' * 10
            # print diagnosis_text
            # if diag.Diag_Arrithmia():
                # print 'XXXX[Is Arrithmia!]XXXX'
            
            debug_count -= 1
            if debug_count <= 0:
                break
    print
    print '#' * 20
    print 'TP:', TP_count
    print 'FN:', FN_count
    print 'FP:', FP_count
    print 'Total:', total_count



    # Get 80% error value
    degree_diffs = [x if x <= 180 else 360 - x for x in degree_diffs]

    with open('./tmp/degree_list.json', 'w') as fout:
        json.dump(degree_diffs, fout)
        print 'degree_diffs written to file ./tmp/degree_list.json.'

    abs_diff_list = [abs(x) if x < 180 else abs(360 - x) for x in degree_diffs]
    abs_diff_list.sort()
    print '80%% difference value: %f' % (abs_diff_list[int(0.8 * len(abs_diff_list))])
    confindence_error_value = abs_diff_list[int(0.8 * len(abs_diff_list))]
    # degree difflist
    plt.hist(degree_diffs, bins = 20, normed = True)
    plt.grid(True)
    plt.title(u'ECG Axis Bias(80%% less than %f) Total %d samples' % (
        confindence_error_value, len(degree_diffs)))
    plt.xlabel(u'Bias(Degree)')
    plt.ylabel(u'Percentage%%')
    # plt.title(u'电轴检测误差(80%% 位于 %f内) 共%d个样本' % (
        # confindence_error_value, len(degree_diffs)))
    # plt.xlabel(u'误差(角度)')
    # plt.ylabel(u'百分比')
    plt.show()
    
def HistTest():
    '''Plot histogram.'''
    with open('./tmp/degree_list.json', 'r') as fin:
        degree_diffs = json.load(fin)
        print 'degree_diffs written to file ./tmp/degree_list.json.'

    abs_diff_list = [abs(x) if x < 180 else abs(360 - x) for x in degree_diffs]
    abs_diff_list.sort()
    print '80%% difference value: %f' % (abs_diff_list[int(0.8 * len(abs_diff_list))])
    confindence_error_value = abs_diff_list[int(0.8 * len(abs_diff_list))]
    # degree difflist
    plt.hist(degree_diffs, bins = 20, normed = True)
    plt.grid(True)
    plt.title(u'ECG Axis Bias(80%% less than %f) Total %d samples' % (
        confindence_error_value, len(degree_diffs)))
    plt.xlabel(u'Bias(Degree)')
    plt.ylabel(u'Percentage%%')
    plt.show()



# HistTest()
# Test()
