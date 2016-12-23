#encoding:utf8
import os
import sys
import bisect
import time
import pickle
import scipy
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import pdb


from DPI_QRS_Detector import DPI_QRS_Detector as DPI
from QTdata.loadQTdata import QTloader
from ecgloader.MITdbLoader import MITdbLoader
from RegressionLearner import RegressionLearner

from HogFeatureExtractor import HogFeatureExtractor
from fast_tester import FastTester


def Test1():
    '''Test function for fast tester.'''
    fast_tester = FastTester()

    mit = MITdbLoader()
    rec_name = mit.getreclist()[11]
    raw_sig = mit.load(rec_name)
    resample_length = int(len(raw_sig) * 250.0 / 360.0)
    raw_sig = scipy.signal.resample(raw_sig, resample_length)
    len_sig = len(raw_sig)

    detected_results = fast_tester.testing(raw_sig)
    
    # plot testing results
    plt.plot(raw_sig, label = 'raw ECG signal')
    labels = set()
    for item in detected_results:
        pos, label = item
        labels.add(label)
        
    for target_label in labels:
        poslist = [x[0] for x in filter(lambda x: x[1] == target_label, detected_results)]
        amp_list = [raw_sig[int(x)] for x in poslist]
        plt.plot(poslist, amp_list, 'o', markersize = 12, label = target_label)
    
    plt.legend()
    plt.show()


# Test_regression()
# Test_hog1d()
Test1()

