import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import ndimage
import argparse
from scipy.optimize import curve_fit


def plot_tuning(responses,freqs, start_pos, coord, start, end):

    coords = np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW'])
    init_param = start_pos[np.where(coords==coord)][0]

    freqs = freqs/10**9 # GHz
    print([freqs[0], freqs[-1], end+init_param, start+init_param])
    plt.imshow(responses, extent=[freqs[0], freqs[-1], end+init_param, start+init_param], interpolation='none', aspect='auto', cmap='plasma_r')
    #plt.imshow(responses, interpolation='none', aspect='auto', cmap='plasma_r')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel(f'Tuning Parameter: {coord[-1]}')
    plt.colorbar()

def fft_cable_ref_filter(responses):

    harmon = 30

    resp_fft = np.fft.rfft(responses, axis=1)

    filted_fft = resp_fft.copy()
    d = 3
    filted_fft[:,harmon-d:harmon+d] = 0
    filted_fft[:,2*harmon] = 0
    filted_resp = np.fft.irfft(filted_fft, n=responses.shape[1])
    
    #plt.imshow(np.abs(filted_fft), aspect='auto', interpolation='none', vmax=1e4)
    #plt.colorbar()
    #plt.figure()

    return filted_resp

def skewedLorentzian(x,bkg,bkg_slp,skw,mintrans,res_f,Q):
    term1 = bkg 
    term2 = bkg_slp*(x-res_f)
    numer = (mintrans+skw*(x-res_f))
    denom = (1+4*Q**2*((x-res_f)/res_f)**2)
    term3 = numer/denom
    return term1 + term2 - term3

def get_lorentz_fit(freqs, spec):

    # define the initial guesses
    bkg = (spec[0]+spec[-1])/2
    bkg_slp = (spec[-1]-spec[0])/(freqs[-1]-freqs[0])
    skw = 0

    mintrans = bkg-spec.min()
    res_f = freqs[spec.argmin()]

    Q = 1e4

    low_bounds = [bkg/2,-1e-3,-1,0,freqs[0],1e2]
    up_bounds = [bkg*2,1e-3,1,30,freqs[-1],1e5]

    popt,pcov = curve_fit(skewedLorentzian,freqs,spec,p0=[bkg,bkg_slp,skw,mintrans,res_f,Q],method='lm')

    return popt


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

    print(responses.shape)

    if i > 0:
        plt.figure()

    #plot_tuning(responses,freqs, start_pos, coord, start, end)
    filted = fft_cable_ref_filter(responses)

    spec = filted[229, 3780:3880]
    
    print(get_lorentz_fit(freqs[3780:3880], spec))

#plt.figure()
#fft_cable_ref_filter(responses,freqs, start_pos, coord, start, end)
plt.show()