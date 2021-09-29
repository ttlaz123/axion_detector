import numpy as np
import matplotlib.pyplot as plt
import re


def plot_tuning(responses,freqs, start_pos, coord, start, end):

    coords = np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW'])
    init_param = start_pos[np.where(coords==coord)][0]

    freqs = freqs/10**9 # GHz
    print([freqs[0], freqs[-1], end+init_param, start+init_param])
    plt.imshow(responses, extent=[freqs[0], freqs[-1], end+init_param, start+init_param], interpolation='none', aspect='auto', cmap='plasma_r')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel(f'Tuning Parameter: {coord[-1]}')
    plt.colorbar()
    plt.show()

plot_dir = "C:\\Users\\FTS\\source\\repos\\axion_detector\\plots\\"
data_dir = "C:\\Users\\FTS\\source\\repos\\axion_detector\\tuning_data\\"

fname = "-0.5479X1.2093Y-0.0184Z0.0099U0.6131V0.6094Wi-0.01f0.01dX.npy"

data = np.load(f"{data_dir}{fname}")
freqs = data[0]
responses = data[1:]

numbers = re.split('[^0-9-.]', fname)
start_pos = np.array(numbers[:6], dtype=float)
start = float(numbers[7])
end = float(numbers[8])
coord = fname[-6:-4]

plot_tuning(responses,freqs, start_pos, coord, start, end)