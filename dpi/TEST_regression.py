#encoding:utf8
import os
import sys
import bisect
import time
import pickle
import scipy
import scipy.signal
import matplotlib.pyplot as plt
import pdb


from DPI_QRS_Detector import DPI_QRS_Detector as DPI
from QTdata.loadQTdata import QTloader
from ecgloader.MITdbLoader import MITdbLoader
from RegressionLearner import RegressionLearner

from HogFeatureExtractor import HogFeatureExtractor

def GetFN(R_pos_list, qrs_list):
    '''Get False Negtives.'''
    MaxGapDistance = 20

    p1 = 0
    p2 = bisect.bisect_left(qrs_list, R_pos_list[0])
    if p2 > 0:
        p2 -= 1

    len_expert = len(R_pos_list)
    len_detect = len(qrs_list)
    
    FN_arr = list()

    is_matched = False
    while p1 < len_expert and p2 < len_detect:
        expert_pos = R_pos_list[p1]
        detect_pos = qrs_list[p2]

        current_dist = abs(expert_pos - detect_pos)
        if current_dist <= MaxGapDistance:
            is_matched = True

        if expert_pos < detect_pos:
            if is_matched:
                is_matched = False
            else:
                FN_arr.append(expert_pos)

            p1 += 1
        else:
            p2 += 1
    return FN_arr

def Test_regression():
    '''Regression test.'''
    target_label = 'T'
    qt = QTloader()
    rec_ind = 65
    reclist = qt.getreclist()
    sig = qt.load(reclist[rec_ind])
    raw_sig = sig['sig']
    len_sig = len(raw_sig)
    
    rf = RegressionLearner(target_label = target_label)
    rf.TrainQtRecords(reclist[0:])
    
    # Load the trained model
    # with open('./tmp.mdl', 'rb') as fin:
        # mdl = pickle.load(fin)
        # rf.LoadModel(mdl)
    # Save the trained model
    with open('./tmp.mdl', 'wb') as fout:
        pickle.dump(rf.mdl, fout)
        
    dpi = DPI()
    qrs_list = dpi.QRS_Detection(raw_sig)

    start_time = time.time()
    detected_poslist = rf.testing(raw_sig, zip(qrs_list, ['R',] * len(qrs_list)))
    print 'Testing time: %d s.' % (time.time() - start_time)
    while len(detected_poslist) > 0 and detected_poslist[-1] > len_sig:
        del detected_poslist[-1]

    sig = qt.load(reclist[rec_ind])
    raw_sig = sig['sig']
    plt.plot(raw_sig, label = 'raw signal')
    amp_list = [raw_sig[int(x)] for x in detected_poslist]
    plt.plot(detected_poslist, amp_list, 'ro', markersize = 12,
             label = target_label)
    plt.legend()
    plt.show()

def Test_hog1d():
    '''Hog feature method test.'''
    target_label = 'T'
    qt = QTloader()
    rec_ind = 103
    reclist = qt.getreclist()
    sig = qt.load(reclist[rec_ind])
    raw_sig = sig['sig']
    len_sig = len(raw_sig)
    
    # Hog feature tester
    hog = HogFeatureExtractor(target_label = target_label)
    rec_list = hog.qt.getreclist()

    training_list = reclist[0:100]
    # training_rec = list(set(rec_list) - set(testing_rec))

    hog.Train(training_list)
    
    # Load the trained model
    # with open('./hog.mdl', 'rb') as fin:
        # mdl = pickle.load(fin)
        # hog.LoadModel(mdl)
    # Save the trained model
    with open('./hog.mdl', 'wb') as fout:
        pickle.dump(hog.gbdt, fout)
        
    dpi = DPI()
    qrs_list = dpi.QRS_Detection(raw_sig)

    start_time = time.time()
    detected_poslist = hog.Testing(raw_sig, zip(qrs_list, ['R',] * len(qrs_list)))
    print 'Testing time: %d s.' % (time.time() - start_time)
    while len(detected_poslist) > 0 and detected_poslist[-1] > len_sig:
        del detected_poslist[-1]
    while len(detected_poslist) > 0 and detected_poslist[0] < 0:
        del detected_poslist[0]

    sig = qt.load(reclist[rec_ind])
    raw_sig = sig['sig']
    plt.plot(raw_sig, label = 'raw signal D1')
    plt.plot(sig['sig2'], label = 'raw signal D2')
    amp_list = [raw_sig[int(x)] for x in detected_poslist]
    plt.plot(detected_poslist, amp_list, 'ro', markersize = 12,
             label = target_label)
    plt.title('Record name %s' % reclist[rec_ind])
    plt.legend()
    plt.show()

def Test_Mit():
    '''Hog feature method test.'''
    target_label = 'T'
    
    # Hog feature tester
    hog = HogFeatureExtractor(target_label = target_label)
    rec_list = hog.qt.getreclist()

    training_list = rec_list[0:]
    # training_rec = list(set(rec_list) - set(testing_rec))

    # hog.Train(training_list)
    
    # Load the trained model
    with open('./hog.mdl', 'rb') as fin:
        mdl = pickle.load(fin)
        hog.LoadModel(mdl)
    # Save the trained model
    # with open('./hog.mdl', 'wb') as fout:
        # pickle.dump(hog.gbdt, fout)
        

    # Testing on mit database
    mit = MITdbLoader()

    rec_name = mit.getreclist()[9]
    raw_sig = mit.load(rec_name)
    resample_length = int(len(raw_sig) * 250.0 / 360.0)
    raw_sig = scipy.signal.resample(raw_sig, resample_length)
    len_sig = len(raw_sig)

    debug_info = dict()
    debug_info['time_cost'] = True
    dpi = DPI(debug_info = debug_info)
    qrs_list = dpi.QRS_Detection(raw_sig)

    start_time = time.time()
    detected_poslist = hog.Testing(raw_sig, zip(qrs_list, ['R',] * len(qrs_list)))
    print 'Testing time: %d s.' % (time.time() - start_time)
    while len(detected_poslist) > 0 and detected_poslist[-1] > len_sig:
        del detected_poslist[-1]
    while len(detected_poslist) > 0 and detected_poslist[0] < 0:
        del detected_poslist[0]

    raw_sig = mit.load(rec_name)
    raw_sig = scipy.signal.resample(raw_sig, resample_length)
    sigd2 = scipy.signal.resample(mit.sigd2, resample_length)

    plt.plot(raw_sig, label = 'raw signal D1')
    plt.plot(sigd2, label = 'raw signal D2')
    amp_list = [raw_sig[int(x)] for x in detected_poslist]
    plt.plot(detected_poslist, amp_list, 'ro', markersize = 12,
             label = target_label)
    plt.title('Record name %s' % rec_name)
    plt.legend()
    plt.show()



# Test_regression()
# Test_hog1d()
Test_Mit()

