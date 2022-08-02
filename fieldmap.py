import numpy as np
import matplotlib.pyplot as plt

import analyse as ana
import na_tracer

from scipy.signal import find_peaks



def main():

    na = na_tracer.NetworkAnalyzer()

    # manually adjust view so that the fundamental is the
    # lowest freq mode in the freq range

    input("Ready to find fiducial frequency. Please move disks out of the way. (ENTER when done)")

    # get resonance baseline
    freq = na.get_pna_freq()
    resp = na.get_pna_complex_response()
    fid_f = ana.get_lowest_trough(freq, resp)

    deltas = np.zeros((2,3,3))
    # front and back, in all nine positions
    x_names = ["front", "rear"]
    y_names = ["left", "middle", "right"]
    z_names = ["top", "middle", "bottom"]

    for i in range(2):
        for j in range(3):
            for k in range(3):
                # for each position, get the freq location and delta
                
                if i == 1 and j == 0 and k == 0:
                    input("Please move the front disk out of the way. (ENTER when done)")

                input(f"Please move {x_names[i]} disk to {z_names[j]}-{y_names[k]}. (ENTER when done)")

                resp = na.get_pna_complex_response()
                pos = ana.get_lowest_trough(freq, resp)
                delta = pos - fid_f
                deltas[i,j,k] = delta


    plt.figure()
    plt.title("Front Map")
    plt.imshow(deltas[0])
    plt.figure()
    plt.title("Rear Map")
    plt.imshow(deltas[1])
    
    plt.show()

if __name__=="__main__":
    main()