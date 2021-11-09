import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import ndimage
import argparse
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import analyse

def plot_tuning(responses,freqs, start_pos, coord, start, end):

    coords = np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW'])
    init_param = start_pos[np.where(coords==coord)][0]

    freqs = freqs/10**9 # GHz
    plt.imshow(responses, extent=[freqs[0], freqs[-1], end+init_param, start+init_param], interpolation='none', aspect='auto', cmap='plasma_r')
    #plt.imshow(responses, interpolation='none', aspect='auto', cmap='plasma_r')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel(f'Tuning Parameter: {coord[-1]}')
    plt.colorbar()

if __name__ == '__main__':

    plot_dir = "C:\\Users\\FTS\\source\\repos\\axion_detector\\plots\\"
    data_dir = "C:\\Users\\FTS\\source\\repos\\axion_detector\\tuning_data\\"

    parser = argparse.ArgumentParser()

    parser.add_argument('fnames', type=str, nargs='+', help=f'data file names to plot (assumed to be in {data_dir} unless --abs specified)')
    parser.add_argument('--abs', dest='data_dir', action='store_const', const='', default=data_dir, help='add to indicate that your fnames have an absolute path')

    args = parser.parse_args()

    data_dir = args.data_dir
    fnames = args.fnames

    for i, fname in enumerate(fnames):

        data = np.load(f"{data_dir}{fname}")
        freqs = data[0]
        responses = data[1:]

        coord_str = fname.split('_')[1]
        numbers = re.split('[^0-9-.]', coord_str)
        start_pos = np.array(numbers[:6], dtype=float)
        start = float(numbers[6])
        end = float(numbers[7])
        coord = fname[-6:-4]

        if i > 0:
            plt.figure()

        N = responses.shape[0]
        incr = (end-start)/N

        specs = analyse.fft_cable_ref_filter(responses, harmon=9)

        coord_num = np.where(coord == np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW']))[0]
        
        '''
        fundamental_inds, skipped, all_inds = analyse.get_fundamental_inds(specs, freqs, search_range=50, search_order='fwd')
        param_space = np.delete(np.linspace(start+incr/2,end-incr/2,N) + start_pos[coord_num], skipped)
        plt.figure()
        plot_tuning(specs, freqs, start_pos, coord, start, end)

        for i,inds in enumerate(all_inds):
            plt.plot(freqs[inds]*1e-9, param_space[i]*np.ones_like(freqs[inds]), 'b.')
        plt.plot(freqs[fundamental_inds]*1e-9, param_space, 'r.')
        #tp = analyse.get_turning_point(specs, 'dX', start_pos, start, end, incr)
        #plt.plot(freqs*1e-9,tp*np.ones_like(freqs))
        plt.figure()
        ind = 8
        plt.plot(freqs,specs[ind])
        plt.plot(freqs[fundamental_inds[ind]], specs[ind][fundamental_inds[ind]], 'r.')
        '''

        plt.figure()
        plot_tuning(specs, freqs, start_pos, coord, start, end)

    '''
    data=np.load(f"{data_dir}{fnames[0]}")
    freqs = data[0]
    spec = data[1]

    popt = analyse.get_lorentz_fit(freqs, spec)
    print(popt)

    plt.plot(freqs,spec, label="spectrum")
    plt.plot(freqs, analyse.skewed_lorentzian(freqs, *popt), label=f"Q={popt[-1]}")
    plt.legend()
    '''

    plt.show()