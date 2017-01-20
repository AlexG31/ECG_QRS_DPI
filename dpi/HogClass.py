#encoding:utf8
import os
import sys
import math
import pdb
import matplotlib.pyplot as plt
import numpy as np


class HogClass(object):
    def __init__(self, segment_len = 100):
        '''
        Inputs:
            segment_len: segment_length of each cell.
        '''
        self.segment_len = segment_len

    def DiscretiseHog(self, sig_in, diff_step = 4, debug_plot = False):
        '''Discretize hog feature.'''
        segment_len = self.segment_len
        
        len_sig = len(sig_in)

        # Hog array
        hog_arr = list()
        dis_hog_arr = list()

        # Baseline array
        h0_arr = list()

        cur_ind = 0

        while cur_ind < len_sig:
            right_ind = min(len_sig - 1, cur_ind + segment_len - 1)

            # Compute diffs in this segment
            diff_arr = [sig_in[di + diff_step] - sig_in[di]
                            for di in xrange(cur_ind, right_ind - diff_step + 1)]
            hog_arr.append(diff_arr)
            h0_arr.append(np.mean(sig_in[cur_ind:right_ind+1]))
            
            # Discretization
            theta_list = [math.atan( float(x * 100.0) / float(diff_step)) for x in diff_arr]
            theta_list.sort()
            
            angle_weights = [0,] * 5
            cur_angle = - 3.0 * math.pi / 8
            cur_angle_ind = 0
            for theta in theta_list:
                if theta < cur_angle + 1e-6:
                    angle_weights[cur_angle_ind] += 1
                else:
                    cur_angle += math.pi / 4.0
                    cur_angle_ind += 1
                    angle_weights[cur_angle_ind] += 1
                    
            dis_hog_arr.append(angle_weights)

            cur_ind += segment_len

        
        # Show Hog
        if debug_plot == True :
            self.VisualizeDiscreteHogArray(dis_hog_arr, h0_arr = h0_arr)

        return dis_hog_arr 

    def ComputeHog(self, sig_in, diff_step = 4, debug_plot = False):
        '''Compute 1-dim hog.'''
        segment_len = self.segment_len
        
        len_sig = len(sig_in)

        # Hog array
        hog_arr = list()

        # Baseline array
        h0_arr = list()

        cur_ind = 0
        while cur_ind < len_sig:
            right_ind = min(len_sig - 1, cur_ind + segment_len - 1)

            # Compute diffs in this segment
            diff_arr = [sig_in[di + diff_step] - sig_in[di]
                            for di in xrange(cur_ind, right_ind - diff_step + 1)]
            hog_arr.append(diff_arr)
            h0_arr.append(np.mean(sig_in[cur_ind:right_ind+1]))
            # plt.hist(diff_arr)
            # plt.show()
            
            cur_ind += segment_len

        # Show Hog
        if debug_plot == True :
            self.VisualizeHogArray(hog_arr, h0_arr = h0_arr)

        return hog_arr

    def GetRealHogArray(self, sig_in, max_norm_degree = 85.0,
            hog_discretize_gap = 5.0, diff_step = 4, debug_plot = False):
        '''Compute hog array with discretization.
        Returns:
            The dicretized hog angles in rad.
        '''
        segment_len = self.segment_len

        # Discretize Ratio
        discretize_ratio = hog_discretize_gap / 180.0 * math.pi
        # print discretize_ratio
        # The maximum possible rad of hog feature
        max_norm_rad = max_norm_degree / 180.0 * math.pi
        max_norm_grad = math.tan(max_norm_rad) * diff_step
        
        len_sig = len(sig_in)

        # Hog array
        hog_arr = list()

        # Baseline array
        h0_arr = list()

        # Largest gradient
        max_grad = 0

        cur_ind = 0
        while cur_ind < len_sig:
            right_ind = min(len_sig - 1, cur_ind + segment_len - 1)

            # Compute diffs in this segment
            diff_arr = [sig_in[di + diff_step] - sig_in[di]
                            for di in xrange(cur_ind, right_ind - diff_step + 1)]

            # print diff_arr
            # plt.figure(1)
            # plt.clf()
            # plt.plot(sig_in)
            # plt.plot([di for di in xrange(cur_ind, right_ind + 1)],
                # [sig_in[di] for di in xrange(cur_ind, right_ind + 1)], 'ro')
            # plt.grid(True)
            # plt.show(block = False)
            # pdb.set_trace()

            max_grad = max(np.max(np.absolute(diff_arr)), max_grad)

            hog_arr.append(diff_arr)
            h0_arr.append(np.mean(sig_in[cur_ind:right_ind+1]))
            
            cur_ind += segment_len

        # Normalize and discretize
        norm_hog_features = list()
        for current_hog in hog_arr:
            current_hog_angles = list()
            for diff in current_hog:
                diff = float(diff) / max_grad * max_norm_grad
                angle_rad = math.atan(diff)
                # Discretize
                angle_rad = float(int(angle_rad / discretize_ratio)) * discretize_ratio
                current_hog_angles.append(angle_rad)
            norm_hog_features.append(current_hog_angles)
            
        # Show Hog
        if debug_plot == True :
            self.VisualizeRealHogArray(norm_hog_features, h0_arr = h0_arr)

        return norm_hog_features

    def VisualizeRealHogArray(self, hog_arr, h0_arr = None):
        '''Visualize hog array of k.'''
        reduce_hog_amplitude_ratio = 1.0 / 100

        segment_len = self.segment_len
        h0 = segment_len / 2
        if h0_arr is None:
            h0_arr = [h0,] * len(hog_arr)

        h0_arr = np.array(h0_arr) * 10.0

        cur_ind = 0
        len_hog_arr = len(hog_arr)

        # Plot
        plt.figure(2)
        for hog_i in xrange(0, len_hog_arr):
            right_ind = cur_ind + segment_len - 1

            # Plot hog direction line
            alpha = 1.0 / len(hog_arr[hog_i])
            x1 = cur_ind
            x2 = right_ind
            for angle_rad in hog_arr[hog_i]:
                k_val = math.tan(angle_rad) * reduce_hog_amplitude_ratio
                y1 = h0_arr[hog_i] - k_val * segment_len / 2.0
                y2 = h0_arr[hog_i] + k_val * segment_len / 2.0
                plt.plot([x1, x2], [y1, y2], alpha = alpha, lw = 2,
                            color = 'black')

            # update cur_ind
            cur_ind += segment_len


    def VisualizeHogArray(self, hog_arr, h0_arr = None):
        '''Visualize hog array of k.'''
        segment_len = self.segment_len
        h0 = segment_len / 2
        if h0_arr is None:
            h0_arr = [h0,] * len(hog_arr)

        h0_arr = np.array(h0_arr) * 10.0

        cur_ind = 0
        len_hog_arr = len(hog_arr)

        # Plot
        plt.figure(2)
        for hog_i in xrange(0, len_hog_arr):
            right_ind = cur_ind + segment_len - 1

            # Plot hog direction line
            alpha = 1.0 / len(hog_arr[hog_i])
            x1 = cur_ind
            x2 = right_ind
            for k_val in hog_arr[hog_i]:
                y1 = h0_arr[hog_i] - k_val * segment_len / 2.0
                y2 = h0_arr[hog_i] + k_val * segment_len / 2.0
                plt.plot([x1, x2], [y1, y2], alpha = alpha, lw = 2,
                            color = 'black')

            # update cur_ind
            cur_ind += segment_len

        # plt.show()
            
    def VisualizeDiscreteHogArray(self, hog_arr, h0_arr = None):
        '''Visualize discrete hog array of k.'''
        segment_len = self.segment_len
        h0 = segment_len / 2
        if h0_arr is None:
            h0_arr = [h0,] * len(hog_arr)

        h0_arr = np.array(h0_arr) * 10.0

        cur_ind = 0
        len_hog_arr = len(hog_arr)

        theta_arr = [-math.pi / 2.0 + x * math.pi / 4.0 for x in xrange(0,5)]
        circle_rad = segment_len / 4.0

        # Plot
        plt.figure(2)
        plt.title('Hog Discrete Visulization')
        for hog_i in xrange(0, len_hog_arr):
            right_ind = cur_ind + segment_len - 1

            # Plot hog direction line
            x1 = (cur_ind + right_ind) / 2.0
            x2 = x1
            for theta, weight in zip(theta_arr, hog_arr[hog_i]):
                if abs(abs(theta) - math.pi / 2.0) == 1e-6:
                    x2 = x1
                else:
                    x2 = x1 + circle_rad * math.cos(theta)
                y1 = h0_arr[hog_i]

                # For visual effect, reduce dy
                y2 = h0_arr[hog_i] + 0.3 * circle_rad * math.sin(theta)


                # print 'hog_i = ', hog_i
                # print 'hog_arr:', hog_arr[hog_i]
                # print 'weight = ', weight
                # pdb.set_trace()

                if sum(hog_arr[hog_i]) == 0:
                    alpha = 0
                else:
                    alpha =  float(weight) / sum(hog_arr[hog_i])

                plt.plot([x1, x2], [y1, y2], alpha = alpha, lw = 6,
                            color = 'black')

            # update cur_ind
            cur_ind += segment_len

        # plt.show()
        
        
        
def Test():
    '''Test function for HOG1D class.'''

    # Import packages in this project
    from QTdata.loadQTdata import QTloader

    qt = QTloader()
    sig_struct  = qt.load('sel17152')
    sig = sig_struct['sig']
    sig = sig[10000:10900]

    # Plot ECG signal
    # plt.figure(1)
    # plt.plot(sig)
    # plt.show()

    # HOG 1d class
    hoger = HogClass(segment_len = 20)
    # hoger.ComputeHog(sig, debug_plot = True)
    hoger.GetRealHogArray(sig, diff_step = 5, debug_plot = True)
    # hoger.DiscretiseHog(sig, debug_plot = True)

    plt.figure(2)
    plt.plot(np.array(sig) * 10)
    plt.grid(True)
    plt.show()

    

if __name__ == '__main__':
    pass
    Test()
            


