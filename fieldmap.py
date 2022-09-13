import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import time

import analyse as ana
import na_tracer
import automate
import datetime

from time import ctime
from scipy.signal import find_peaks
from scipy.special import expit

def main():

    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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

    #print("Autoaligning")
    #automate.autoalign(auto, coords, margins, coarse_ranges=np.zeros(3), fine_ranges=fine_ranges, search_orders=search_orders, harmon=harmon, skip_coarse=skip_coarse)

    # get resonance baseline
    #time.sleep(1)
    N = 5
    fid_f = 0
    for i in range(N):
        
        resp = na.get_pna_response()

        fid_f += ana.get_lowest_trough(freq, resp)

    fid_f /= N

    # number of points in each direction
    resy = 1
    resz = 1

    deltas = np.zeros((2,resz,resy)) #this is hardcoded for grid pattern

    # front and back, in all positions
    x_names = ["front", "rear"]
    #y_names = ["left", "middle", "right"]
    #z_names = ["top", "middle", "bottom"]
    #will tell you to move to 2 numbers: (height, lateral)

    for i in range(2):
        for j in range(resz): #thought it might be better to not hardcode 3x3 pattern in case we want other patterns
            for k in range(resy):
                #for each postiion, get the freq location and delta
                
                if i == 1 and j == 0 and k == 0:
                    input("Please move the front disk out of the way. (ENTER when done)")
                
                #depending on if z is even or odd we want to go in a different across direction to create snake
                if j % 2 == 0: #sweep right to left
                    # (row, col)
                    input(f"Please move {x_names[i]} disk to ({j+1},{resy-k}). (ENTER when done)")
                
                if j % 2 == 1: #sweep left to right
                    input(f"Please move {x_names[i]} disk to ({j+1},{k+1}). (ENTER when done)")
                
                #print("Autoaligning")
                #automate.autoalign(auto, coords, margins, fine_ranges=fine_ranges, search_orders=search_orders, harmon=harmon, skip_coarse=skip_coarse)
                #chao-lin said let's not autoalign after every step, move back to corner to autoalign
                #record hexapod locations at every step
                #time.sleep(1)
                resp = na.get_pna_response()
                pos = ana.get_lowest_trough(freq, resp)
                delta = pos - fid_f

                if j%2 == 0:
                    deltas[i,j,-1-k] = delta
                if j%2 != 0:
                    deltas[i,j,k] = delta


    # for i in range(2):
    #     for j in range(3):
    #         for k in range(3):
    #             # for each position, get the freq location and delta
                
    #             if i == 1 and j == 0 and k == 0:
    #                 input("Please move the front disk out of the way. (ENTER when done)")

    #             input(f"Please move {x_names[i]} disk to {z_names[j]}-{y_names[k]}. (ENTER when done)")

    #             print("Autoaligning")
    #             automate.autoalign(auto, coords, margins, fine_ranges=fine_ranges, search_orders=search_orders, harmon=harmon, skip_coarse=skip_coarse)

    #             time.sleep(1)
    #             resp = na.get_pna_response()
    #             pos = ana.get_lowest_trough(freq, resp)
    #             delta = pos - fid_f
    #             deltas[i,j,k] = delta

    '''
    hist_tot, edges_tot = np.histogram(deltas, bins="auto")
    hist_front, edges_front = np.histogram(deltas[0], bins="auto")
    hist_back, edges_back = np.histogram(deltas[1], bins="auto")
    '''

    err, hexa_position = hexa.get_position()
    coord_names = ['X', 'Y', 'Z', 'U', 'V', 'W']

    pos_dict = dict(zip(coord_names, hexa_position))

    with open(f"field_mapping_data\\{filename}.json", "w") as write_file:
        json.dump(pos_dict, write_file)

    np.savetxt(f'field_mapping_data\\{filename}.csv', np.hstack((fid_f, deltas.flatten())))
    
    bin_edges = np.linspace(np.min(deltas), np.max(deltas), 20)
    plt.figure()
    plt.hist(deltas.flatten(), bins=bin_edges, label="total", lw = 1, fc=(0, 0, 0, 0.5))
    plt.hist(deltas[0].flatten(), bins=bin_edges, label="front", lw = 1, fc=(0, 0, 1, 0.5))
    plt.hist(deltas[1].flatten(), bins=bin_edges, label="back", lw = 1, fc=(1, 0, 0, 0.5))
    plt.xlabel(f"$\\Delta f$ (Hz)")
    plt.legend()
    plt.savefig(f'field_mapping_plots\\hist_{filename}.png')
    plt.figure()
    plt.title("Front Map")
    plt.imshow(deltas[0])
    plt.colorbar(label = 'Hz')
    plt.savefig(f'field_mapping_plots\\front_{filename}.png')
    plt.figure()
    plt.title("Rear Map")
    plt.imshow(deltas[1])
    plt.colorbar(label='Hz')
    plt.savefig(f'field_mapping_plots\\rear_{filename}.png')
    
    plt.show()

if __name__=="__main__":
    main()