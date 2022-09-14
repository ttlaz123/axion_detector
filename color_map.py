"""
Produces a color map representing the resonant frequency shift caused by foam disk movements.
Intakes file produced by fieldmap.py with the center frequency and shift magnitudes (Hz).

Ashley Davidson
Physics Summer Research
August 4, 2022
"""

import matplotlib.pyplot as plt
import numpy as np
import math

FILENAME = 'C:/Users/FTS/source/repos/axion_detector/field_mapping_data/20220913_162645.csv'

def read_deltas(file, return_fres=False):
    with open(file) as f:
        vals = f.readlines()
        #this works provided the measurements were taken in a square:
        dim = int(math.sqrt((len(vals) - 1) // 2 ))
        deltas = np.zeros((2, dim, dim))
        counter = 0
        for i in range(2):
            for j in range(dim):
                for k in range(dim):
                    counter += 1
                    deltas[i, j, k] = float(vals[counter])
        print(str(deltas))
        if return_fres:
            fres = float(vals[0])
            retval = fres, deltas
        else:
            retval = deltas
        return retval

def plot_deltas(deltas):
    plt.figure()
    plt.title("Front Map")
    plt.imshow(deltas[0])
    plt.colorbar(label='Resonance Shift (Hz)')
    #plt.savefig(f'field_mapping_plots\\front_{filename}.png')
    plt.figure()
    plt.title("Rear Map")
    plt.imshow(deltas[1])
    plt.colorbar(label='Resonance Shift (Hz)')
    #plt.savefig(f'field_mapping_plots\\rear_{filename}.png')
    plt.show()

def plot_Es(deltas, mirror_rear=False):
    Es = np.sqrt(-1*deltas)
    plt.figure()
    plt.title("Front Map")
    plt.imshow(Es[0], aspect='auto', interpolation='none', cmap='Spectral_r')
    plt.colorbar(label='E Field (arb. units)')
    #plt.savefig(f'field_mapping_plots\\front_{filename}.png')
    if mirror_rear:
        Es[1] = np.flip(Es[1], axis=1)
    plt.figure()
    plt.title("Rear Map")
    plt.imshow(Es[1], aspect='auto', interpolation='none', cmap='Spectral_r')
    plt.colorbar(label='E Field (arb. units)')
    #plt.savefig(f'field_mapping_plots\\rear_{filename}.png')

def plot_hists(deltas):
    bin_edges = np.linspace(np.min(deltas), np.max(deltas), 10)
    plt.figure()
    plt.hist(deltas.flatten(), bins=bin_edges, label="total", edgecolor="black",lw = 3, fc=(0, 0, 0, 0.5))
    plt.hist(deltas[0].flatten(), bins=bin_edges, label="front", edgecolor="black",lw = 3, fc=(0, 0, 1, 0.5))
    plt.hist(deltas[1].flatten(), bins=bin_edges, label="back", edgecolor="black",lw = 3, fc=(1, 0, 0, 0.5))
    plt.xlabel(f"$\\Delta f$ (Hz)")
    plt.legend()
    plt.show()


def main():
    deltas = read_deltas(FILENAME)
    plot_Es(deltas, mirror_rear=True)
    plt.show()

if __name__ == '__main__':
    main()