import numpy as np
import matplotlib.pyplot as plt
import re
from scipy import ndimage
import argparse


def plot_tuning(responses,freqs, start_pos, coord, start, end):

    coords = np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW'])
    init_param = start_pos[np.where(coords==coord)][0]

    freqs = freqs/10**9 # GHz
    print([freqs[0], freqs[-1], end+init_param, start+init_param])
    plt.imshow(responses, extent=[freqs[0], freqs[-1], end+init_param, start+init_param], interpolation='none', aspect='auto', cmap='plasma_r')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel(f'Tuning Parameter: {coord[-1]}')
    plt.colorbar()

def fft_cable_ref_filter(responses,freqs, start_pos, coord, start, end):

    resp_fft = np.fft.rfft(responses, axis=1)
    spec = np.median(np.abs(resp_fft), axis=0)
    #plt.plot(spec, '.-')
    medfilted_fft = ndimage.median_filter(spec, size=7)
    #plt.plot(medfilted_fft)
    #plt.figure()
    filted_fft = resp_fft.copy()
    d = 3
    filted_fft[:,30] = 0
    filted_fft[:,60] = 0
    filted_fft1 = resp_fft.copy()
    filted_fft1[:,30] = 0
    filted_fft1[:,60] = 0
    filted_resp = np.fft.irfft(filted_fft)
    filted_resp1 = np.fft.irfft(filted_fft1)
    #plot_tuning(responses,freqs, start_pos, coord, start, end)
    #plt.figure()
    #plot_tuning(np.abs(resp_fft),freqs, start_pos, coord, start, end)
    #plt.figure()
    plt.imshow(np.abs(filted_fft), aspect='auto', interpolation='none', vmax=1e4)
    plt.colorbar()
    plt.figure()
    plot_tuning(filted_resp,freqs, start_pos, coord, start, end)


#parser = argparse.ArgumentParser()

#parser.add_argument('fnames', )


plot_dir = "C:\\Users\\FTS\\source\\repos\\axion_detector\\plots\\"
data_dir = "C:\\Users\\FTS\\source\\repos\\axion_detector\\tuning_data\\"

fname = "2021-10-06-17-42-47_0.6087336076081X-1.366887032529Y-8.53988131288Z-0.1616771618321U0.7138277745479V0.6347092183374W-0.5i0.5fdX.npy"

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

plot_tuning(responses,freqs, start_pos, coord, start, end)
#plt.figure()
#fft_cable_ref_filter(responses,freqs, start_pos, coord, start, end)
plt.show()