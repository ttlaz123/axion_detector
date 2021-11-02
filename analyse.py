import numpy as np
import matplotlib.pyplot as plt

import tuning_plotter

from scipy.signal import find_peaks

def fft_cable_ref_filter(responses, harmon=30):

    resp_fft = np.fft.rfft(responses, axis=1)

    filted_fft = resp_fft.copy()
    d = 3
    filted_fft[:,harmon-d:harmon+d] = 0
    filted_fft[:,2*harmon] = 0
    filted_resp = np.fft.irfft(filted_fft, n=responses.shape[1])
    
    #plt.figure()
    #plt.imshow(np.abs(filted_fft), aspect='auto', interpolation='none', vmax=1e4)
    #plt.colorbar()
    #plt.figure()

    return filted_resp

def skewed_lorentzian(x,bkg,bkg_slp,skw,mintrans,res_f,Q):
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

    popt,pcov = curve_fit(skewed_lorentzian,freqs,spec,p0=[bkg,bkg_slp,skw,mintrans,res_f,Q],method='lm')

    return popt

def get_fundamental_inds(responses, search_order='fwd'):
    '''
    fit an equation to the fundamental mode on a mode map

    start_from is either fwd or rev. We look for peaks near the one just found, but 
    we need to start somewhere. So, we find the leftmost (lowest freq) peak on either the
    top or bottom row, working either top down (fwd) or bottom up (rev).
    Usually the resonance is strongest near the top row, so fwd is default.
    '''
    N = responses.shape[0]
    fundamental_inds = np.zeros_like(responses[:,0], dtype=int)

    if search_order == 'fwd':
        pass
    elif search_order == 'rev':
        responses = np.flip(responses,axis=0)
    else:
        print('get_fundamental got a bad arg for search_order, exiting')
        exit(-1)

    last_peak_pos = 0 # looking for lowest freq peak first
    bounds_start = 0
    bounds_end = responses[0].size-1
    search_range = 175 # after first iter, number of points on each side to look for next peak

    initial_prominence = 2
    subsequent_prominence = 0.25
    for i,spec in enumerate(responses):
        if i == 0:
            prominence = initial_prominence
        else:
            prominence = subsequent_prominence
        peaks, _ = find_peaks(-spec[bounds_start:bounds_end], width=[0,130], prominence=prominence)
        if len(peaks) == 0:
            fundamental_inds[i] = -1
            # can be smart about where next search range should be
            continue
        peak_num = np.argmin(abs(peaks-last_peak_pos)) # find peak closest to previous one
        fundamental_inds[i] = peaks[peak_num] + bounds_start # must correct for search range limits
        last_peak_pos = fundamental_inds[i]
        bounds_start = last_peak_pos - search_range
        bounds_end = last_peak_pos + search_range

    # skip peak if not found
    skipped = np.where(fundamental_inds < 0)
    fundamental_inds = np.delete(fundamental_inds, skipped)
    '''
    param_space = np.linspace(start+incr/2,end-incr/2,N) + start_pos[0]
    plt.plot(freqs[fundamental_inds]*1e-9, param_space, 'r.')
    plot_tuning(responses, freqs, start_pos, coord, start, end)
    plt.figure()
    ind = 11
    plt.plot(freqs,responses[ind])
    plt.plot(freqs[fundamental_inds[ind]], responses[ind][fundamental_inds[ind]], 'r.')
    '''
    return fundamental_inds, skipped

def get_turning_point(responses, coord, start_pos, start, end, incr, freqs,plot=False):

    coord_num = np.where(np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW']) == coord)[0]

    fundamental_inds, skipped = get_fundamental_inds(responses)
    N = responses.shape[0]
    x = np.linspace(start+incr/2,end-incr/2,N) + start_pos[coord_num]
    x = np.delete(x, skipped)
    y = freqs[fundamental_inds]*1e-9

    p = np.polyfit(x, y, deg=2) # highest degree first in p

    turning_point = -p[1]/(2*p[0])

    if plot:
        plt.figure()
        tuning_plotter.plot_tuning(responses, freqs, start_pos, coord, start, end)
        plt.plot(y, x, 'r.')
        plt.plot(np.polyval(p,x), x, 'b--')
        plt.plot(freqs*1e-9,turning_point*np.ones_like(freqs), 'b')
        plt.plot(freqs*1e-9,start_pos[coord_num]*np.ones_like(freqs), 'k--')

    return turning_point