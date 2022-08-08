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

FILENAME = '/Users/AshleyDavidson/Downloads/1659648673'


def read_data(file):
    with open(file) as f:
        vals = f.readlines()
        #this works provided the measurements were taken in a square pattern:
        dim = int(math.sqrt((len(vals) - 1) // 2 ))
        deltas = np.zeros((2, dim, dim))
        counter = 0
        for i in range(2):
            for j in range(dim):
                for k in range(dim):
                    counter += 1
                    deltas[i, j, k] = float(vals[counter])
        print(str(deltas))
        return deltas

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

def main():
    deltas = read_data(FILENAME)
    plot_deltas(deltas)

if __name__ == '__main__':
    main()