import numpy as np
import matplotlib.pyplot as plt
import time
import argparse

import analyse as ana
import na_tracer
import automate

from scipy.signal import find_peaks
from scipy.special import expit

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--hex_ip', default='192.168.254.254',
                    help='IP address to connect to the NewportXPS hexapod')
    parser.add_argument('-p', '--hex_password', help='Password to connect to the NewportXPS hexapod')
    args = parser.parse_args()

    na = na_tracer.NetworkAnalyzer()

    pos = None
    webhook = None
    hexa = automate.HexaChamber(host=args.hex_ip, username='Administrator', password=args.hex_password)

    auto = automate.AutoScanner(hexa, pos, na, webhook)

    # autoalign params
    coords = ['dX', 'dV', 'dW']
    margins = [0.01,0.01,0.01]
    fine_ranges = np.array([0.02,0.05,0.05])
    search_orders = ['fwd','fwd','rev']
    skip_coarse = True
    save = False

    # get the harmonic to remove to mitigate cable refs.
    freq = na.get_pna_freq()
    _, harmon = ana.auto_filter(freq, np.zeros(9), return_harmon=True) # dummy argument for the actual spectrum

    input("Ready to find fiducial frequency. Please move disks out of the way and adjust the VNA so that the desired reonance is the only in view. (ENTER when done)")

    print("Autoaligning")
    automate.autoalign(auto, coords, margins, coarse_ranges=np.zeros(3), fine_ranges=fine_ranges, search_orders=search_orders, harmon=harmon, skip_coarse=skip_coarse)

    # get resonance baseline
    time.sleep(1)
    resp = na.get_pna_response()

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

                print("Autoaligning")
                automate.autoalign(auto, coords, margins, fine_ranges=fine_ranges, search_orders=search_orders, harmon=harmon, skip_coarse=skip_coarse)

                time.sleep(1)
                resp = na.get_pna_response()
                pos = ana.get_lowest_trough(freq, resp)
                delta = pos - fid_f
                deltas[i,j,k] = delta


    time0 = time.time()
    np.save(f'field_mapping_data\\{time0}', np.hstack((fid_f, deltas.flatten())))
    plt.figure()
    plt.title("Front Map")
    plt.imshow(deltas[0])
    plt.figure()
    plt.title("Rear Map")
    plt.imshow(deltas[1])
    
    plt.show()

if __name__=="__main__":
    main()