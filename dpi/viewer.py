#encoding
import matplotlib.pyplot as plt
import pdb
import os


def grid_data(grid_x, grid_y, max_x, x_step, max_amplitude, y_step):
    for row_ind in xrange(0, max_x, x_step):
        start_y = -max_amplitude
        for y_ind in xrange(0, int(2 * max_amplitude / y_step)):
            y_down = start_y
            start_y += y_step
            y_up = start_y
            x_left = row_ind
            x_right = x_left + x_step
            grid_x.append(x_left)
            grid_y.append(y_up)

            grid_x.append(x_right)
            grid_y.append(y_up)

            grid_x.append(x_right)
            grid_y.append(y_down)
        
            grid_x.append(x_left)
            grid_y.append(y_down)

        grid_x.append(row_ind)
        grid_y.append(-max_amplitude)
        grid_x.append(row_ind + x_step)
        grid_y.append(-max_amplitude)

def plot(raw_sig, fs, max_amplitude):
    '''Plot Standard ECG Diagram.'''
    plt.figure(1)
    plt.clf()

    poslist = [float(x) / fs * 1000.0 for x in xrange(0, len(raw_sig))]
    plt.plot(poslist, raw_sig, lw = 2, color = 'black', label = 'ECG')

    # Plot grids
    grid_x = list()
    grid_y = list()
    grid_data(grid_x, grid_y, int(1000.0 * len(raw_sig) / fs + 200), 200,
            max_amplitude, max_amplitude / 2.0)

    plt.plot(grid_x, grid_y, 'r', alpha = 0.3, lw = 2, label = 'ECG')

    grid_x = list()
    grid_y = list()
    grid_data(grid_x, grid_y, int(1000.0 * len(raw_sig) / fs + 200), 40,
            max_amplitude, max_amplitude / 10.0)
    plt.plot(grid_x, grid_y, 'r--', alpha = 0.2, lw = 1, label = 'ECG')

    plt.title('ECG')
    plt.grid(True, which = 'major')
    plt.legend()
    plt.xlabel('Ms')
    plt.ylabel('Voltage(mV)')
    plt.ylim((-max_amplitude, max_amplitude))
    plt.xlim((0,1000))
    plt.show(block = False)
    pdb.set_trace()

def save_fig(raw_sig, fs, max_amplitude, fname):
    '''Save Ecg segments to file.'''
    
    filepath_parts = os.path.split(fname)
    img_folder = filepath_parts[0]
    img_fname = filepath_parts[-1]

    filename_parts = img_fname.split('.')
    
    part_ind = 1
    max_x = int(1000.0 * len(raw_sig) / fs + 201)
    while True:
        if (part_ind - 1) * 1000 >= max_x:
            break
        x_range = [(part_ind - 1) * 1000, part_ind * 1000]
        plt.figure(1)
        plt.clf()

        poslist = [float(x) / fs * 1000.0 for x in xrange(0, len(raw_sig))]
        plt.plot(poslist, raw_sig, lw = 2, color = 'black', label = 'ECG')

        # Plot grids
        grid_x = list()
        grid_y = list()
        grid_data(grid_x, grid_y, max_x, 200,
                max_amplitude, max_amplitude / 2.0)

        plt.plot(grid_x, grid_y, 'r', alpha = 0.3, lw = 2, label = 'ECG')

        grid_x = list()
        grid_y = list()
        grid_data(grid_x, grid_y, max_x, 40,
                max_amplitude, max_amplitude / 10.0)
        plt.plot(grid_x, grid_y, 'r--', alpha = 0.2, lw = 1, label = 'ECG')

        plt.title('ECG')
        plt.grid(True, which = 'major')
        plt.legend()
        plt.xlabel('Ms')
        plt.ylabel('Voltage(mV)')
        plt.ylim((-max_amplitude, max_amplitude))
        plt.xlim(x_range)
        plt.show(block = False)
        part_name = os.path.join(img_folder,
                filename_parts[0] + '_part%d' % part_ind + '.' + filename_parts[1])
        plt.savefig(part_name)
        part_ind += 1
    

if __name__ == '__main__':
    from QTdata.loadQTdata import QTloader
    qt = QTloader()
    sig = qt.load('sel100')
    raw_sig = sig['sig'][0:10000]
    # plot(raw_sig, 250, 2)
    save_fig(raw_sig, 250, 2, '/tmp/tmp.png')
