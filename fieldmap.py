import numpy as np
import matplotlib.pyplot as plt

import analyse as ana
import na_tracer

from scipy.signal import find_peaks

def auto_filter(response, plot=False):

    ffted = np.fft.fft(response)

    peaks, properties = find_peaks(ffted, width=[0,10], height=100)


    plt.plot(response)
    plt.figure()
    x = np.arange(ffted.size)
    plt.plot(x, ffted)
    plt.plot(x[peaks], ffted[peaks], 'r.')
    plt.show()


def get_lowest_trough(response):
    pass

def main():

    na = na_tracer.NetworkAnalyzer()

    # get resonance baseline

    deltas = np.zeros((2,3,3))
    # front and back, in all nine positions

    # for each position, get the freq location and delta

    freq = na.get_pna_freq()
    resp = na.get_pna_response()

    auto_filter(resp)
    # find lowest freq peak (steal from autoalign)
    # save to some array
    # plot and save

if __name__=="__main__":
    main()