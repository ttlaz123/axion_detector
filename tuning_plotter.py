import numpy as np
import matplotlib.pyplot as plt
import re
import argparse
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import glob

import analyse

def plot_tuning(responses,freqs, start_pos, coord, start, end):

    coords = np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW'])
    init_param = start_pos[np.where(coords==coord)][0]

    freqs = freqs/10**9 # GHz
    plt.imshow(responses, extent=[freqs[0], freqs[-1], end+init_param, start+init_param], interpolation='none', aspect='auto', cmap='plasma_r')
    #plt.imshow(responses, interpolation='none', aspect='auto', cmap='plasma_r')
    plt.title(f"Mode Map for {coord[-1]}", fontsize=30, y=1.01)
    plt.xlabel('Frequency (GHz)', fontsize=20)
    plt.ylabel(f'{coord[-1]} Position (mm)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    cb = plt.colorbar()
    cb.set_label("S11 (dB)", fontsize=20)
    cb.ax.tick_params(labelsize=20)

def load_tuning(fname):

    data = np.load(f"{data_dir}{fname}")
    freqs = data[0]
    responses = data[1:]

    coord_str = fname.split('_')[1]
    numbers = re.split('[^0-9-.]', coord_str)
    start_pos = np.array(numbers[:6], dtype=float)
    start = float(numbers[6])
    end = float(numbers[7])
    coord = fname[-6:-4]

    return freqs, responses, start_pos, coord, start, end

def plot_dir_with_spectra(spec_dir):

    # this is for over-plotting all single spectra in a directory & fitting fundamental

    fnames = np.array(glob.glob(rf'{spec_dir}/*.npy'))
    Zposs = np.zeros_like(fnames, dtype=float)

    # darn arbitrary ordering!
    for i,fname in enumerate(fnames):
        Zposs[i] = fname[-9:-5]

    inds = np.argsort(Zposs)
    fnames = fnames[inds]
    Zposs = Zposs[inds]

    s = 0
    e = -1
    plt.figure(figsize=(12,8))
    for i, fname in enumerate(fnames):
        print(f'reading: {fname}')

        Zpos = Zposs[i]
        
        freqs, response = np.load(fname)

        freqs = freqs[s:e]
        response = response[s:e]

        peaks, properties = find_peaks(-response, prominence=3, width=1)

        fund_peak = peaks[0]

        width = int(np.round(properties['widths'][0] * 2))
        left = fund_peak - width
        right = fund_peak + width

        fit_freqs = freqs[left:right]
        fit_response = response[left:right]

        popt = analyse.get_lorentz_fit(fit_freqs, fit_response)
        
        plt.title("Fundamental Mode at Several Tuning Positions", fontsize=30)
        plt.xlabel("Frequency (GHz)", fontsize=20)
        plt.ylabel("S11 (dB)", fontsize=20)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.plot(freqs*1e-9, response, label=f"Height: {int(Zpos-Zposs[0])} mm, Q: {int(np.round(popt[-1]))}")
        #plt.plot(freqs[fund_peak]*1e-9, response[fund_peak], 'ro')
        plt.plot(fit_freqs*1e-9, analyse.skewed_lorentzian(fit_freqs, *popt), 'k--')
        
    plt.legend(fontsize=15)

def plot_single_spectrum(fname, start=0, end=-1):
    freqs, response = np.load(f'{data_dir}{fname}')

    s = 2950
    e=3125

    freqs = freqs[s:e]

    spec = analyse.fft_cable_ref_filter(response, harmon=9)

    spec=spec[s:e]

    popt = analyse.get_lorentz_fit(freqs, spec)

    plt.plot(freqs*1e-9,spec, 'k')
    plt.plot(freqs*1e-9,analyse.skewed_lorentzian(freqs,*popt), 'r', label=f"Q: {popt[-1]}")
    plt.title('Spectrum of Axion Cavity')
    plt.ylabel('S11 (dB)')
    plt.xlabel('Frequency (GHz)')
    plt.legend()


if __name__ == '__main__':

    plot_dir = "C:\\Users\\FTS\\source\\repos\\axion_detector\\plots\\"
    data_dir = "C:\\Users\\FTS\\source\\repos\\axion_detector\\tuning_data\\"
    #data_dir = "/home/tdyson/coding/axion_detector/tuning_data/"

    parser = argparse.ArgumentParser()

    parser.add_argument('fnames', type=str, nargs='+', help=f'data file names to plot (assumed to be in {data_dir} unless --abs specified)')
    parser.add_argument('--abs', dest='data_dir', action='store_const', const='', default=data_dir, help='add to indicate that your fnames have an absolute path')

    args = parser.parse_args()

    data_dir = args.data_dir
    fnames = args.fnames

    for i, fname in enumerate(fnames):

        freqs, responses, start_pos, coord, start, end = load_tuning(fname)
        
        if i > 0:
            plt.figure()

        N = responses.shape[0]
        incr = (end-start)/N

        specs = analyse.fft_cable_ref_filter(responses, harmon=64)

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

        plt.figure(figsize=(12,8))
        plot_tuning(specs, freqs, start_pos, coord, start, end)
        plt.figure()
        plot_tuning(responses, freqs, start_pos, coord, start, end)

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
