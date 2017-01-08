#encoding:utf8


import os
import sys
import bisect
import time
import pickle
import scipy
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pdb


from DPI_QRS_Detector import DPI_QRS_Detector as DPI
# from QTdata.loadQTdata import QTloader
from HogFeatureExtractor import HogFeatureExtractor

class FastTester():
    def __init__(self):
        '''Fast testing of ECG characteristic points.'''

        # Train models for 'T', 'Toffset', 'Ponset', 'P', 'Poffset' labels
        self.model_dict = dict()

        # for target_label in ['T', 'Toffset', 'Ponset', 'P', 'Poffset']:
            # # Hog feature tester
            # hog = HogFeatureExtractor(target_label = target_label)
            # rec_list = hog.qt.getreclist()

            # hog.Train(rec_list)
            # self.model_dict[target_label] = hog

        # with open('./FastTester.mdl', 'wb') as fout:
            # pickle.dump(self.model_dict, fout)
        with open('FastTester.mdl', 'rb') as fin:
            self.model_dict = pickle.load(fin)

        

    def testing(self, raw_sig, fs = 250.0):
        '''Testing API.
        Returns:
            A list of (index, label) pairs. For example:
            [(1, 'R'), (24, 'T'), (35, 'Toffset'),]
        '''
        if isinstance(raw_sig, list):
            len_sig = len(raw_sig)
        elif isinstance(raw_sig, np.ndarray):
            len_sig = raw_sig.size

        detected_results = list()

        # Detect R first
        debug_info = dict()
        debug_info['time_cost'] = True
        dpi = DPI(debug_info = debug_info)
        qrs_list = dpi.QRS_Detection(raw_sig, fs = fs)
        detected_results.extend(zip(qrs_list, ['R',] * len(qrs_list)))
        
        for target_label in ['T', 'Toffset', 'Ponset', 'P', 'Poffset']:
            start_time = time.time()
            detected_poslist = self.model_dict[target_label].Testing(raw_sig, detected_results)
            print 'Testing time: %d s.' % (time.time() - start_time)
            while len(detected_poslist) > 0 and detected_poslist[-1] > len_sig:
                del detected_poslist[-1]
            while len(detected_poslist) > 0 and detected_poslist[0] < 0:
                del detected_poslist[0]


            detected_results.extend(zip(detected_poslist, [target_label, ] * len(detected_poslist)))
        return detected_results


def test0():
    '''Test code for fast_tester.'''
    from QTdata.loadQTdata import QTloader
    qt = QTloader()
    sig = qt.load('sel100')

    range_right = 1000
    raw_sig = sig['sig2'][0:range_right]

    ft = FastTester()
    res_list = ft.testing(raw_sig, fs = 250.0)
    labels = set([x[1] for x in res_list])
    plt.plot(raw_sig)
    for label in labels:
        poslist = [x[0] for x in filter(lambda x: x[1] == label, res_list)]
        amplist = [raw_sig[int(x)] for x in filter(lambda x: x < range_right, poslist)] 
        plt.plot(poslist, amplist, 'o', markersize = 12, alpha = 0.5, label = label)
    plt.title('sel100')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test0()
    
    
        

