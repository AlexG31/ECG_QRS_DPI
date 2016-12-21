#encoding:utf8
import os
import sys
import bisect
import matplotlib.pyplot as plt


from DPI_QRS_Detector import DPI_QRS_Detector as DPI
from QTdata.loadQTdata import QTloader

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

def Test1():
    '''Comparing to expert labels in QTdb.'''
    qt = QTloader()
    reclist = qt.getreclist()

    rec_ind = 0
    for rec_ind in xrange(0, len(reclist)):

        print 'Processing record[%d] %s ...' % (rec_ind, reclist[rec_ind])
        sig = qt.load(reclist[rec_ind])
        raw_sig = sig['sig']
        expert_labels = qt.getExpert(reclist[rec_ind])
        R_pos_list = [x[0] for x in filter(lambda item: item[1] == 'R', expert_labels)]

        # Skip empty expert lists
        if len(R_pos_list) == 0:
            continue

        dpi = DPI()
        
        qrs_list = dpi.QRS_Detection(raw_sig)

        # Find FN
        FN_arr = GetFN(R_pos_list, qrs_list)
        R_pos_list = FN_arr
        
        if len(R_pos_list) > 0:
            plt.plot(raw_sig)
            amp_list = [raw_sig[x] for x in qrs_list]
            plt.plot(qrs_list, amp_list, 'ro', markersize = 12)
            amp_list = [raw_sig[x] for x in R_pos_list]
            plt.plot(R_pos_list, amp_list, 'ys', markersize = 14)
            plt.show()


Test1()

