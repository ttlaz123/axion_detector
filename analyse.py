import numpy as np
import matplotlib.pyplot as plt
import time

import tuning_plotter

from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties.umath import *
import argparse

def fft_cable_ref_filter(responses, harmon=9, plot=False):

    print(responses)

    if len(responses.shape) == 1:
        resp_fft = np.fft.rfft(responses)
    else:
        resp_fft = np.fft.rfft(responses, axis=1)
    filted_fft = resp_fft.copy()
    if len(responses.shape) == 1:
        filted_fft[harmon-1:harmon+2] = 0
        filted_fft[2*harmon-1:2*harmon+2] = 0
    else:
        filted_fft[:,harmon-1:harmon+2] = 0
        filted_fft[:,2*harmon-1:2*harmon+2] = 0
        filted_fft[:,:10] = 0

    filted_resp = np.fft.irfft(filted_fft, n=responses.shape[-1])
        
    if plot:
        plt.figure()
        if len(responses.shape) == 1:
            plt.plot(filted_fft)
        else:
            plt.imshow(np.abs(filted_fft), aspect='auto', interpolation='none', vmax=1e4)
            plt.colorbar()

        plt.show()
    return filted_resp

def auto_filter(freq, response, a=3e-8, rng=1, plot=False, return_harmon=False):
    """
    Calculates where the cable reflection peak should be based on 
    the frequency range, and removes modes around there.

    a: the proportionality constant between freq range and index in fft
    of the cable reflection signal (measure this by looking at the fft,
    and seeing which peak you need to remove to get rid of the refs. then
    divide that ind by the span of the VNA for that measurement to get "a")
    rng: number of indices to set to zero on each side of the target
    """

    ffted = np.fft.rfft(response)
    ind = int(np.round((freq[-1]-freq[0])*a))
    ffted[ind-rng:ind+rng] = 0
    filted = np.fft.irfft(ffted)

    if plot:
        plt.figure()
        plt.plot(response)

        plt.plot(filted)
        plt.figure()
        plt.plot(np.abs(ffted))

    if return_harmon:
        retval = (filted, ind)
    else:
        retval = filted
    
    return retval

def get_lowest_trough(freq, response):
    """
    Does peak finding and lorentz fitting to find the location of the lowest frequency
    peak in a single spectrum (for field mapping)
    """

    filted = auto_filter(freq, response) #needs both freq and resp? not just resp?

    freq_window = 0.0015*1e9 #measured val in GHz of about how wide half the peak is
    freq_step = freq[1]-freq[0]
    pts_window = int(np.round(freq_window / freq_step))

    idx = response.argmin()
    lorentz_freq = freq[idx - pts_window : idx + pts_window]
    lorentz_resp = response[idx - pts_window : idx + pts_window]
    '''
    plt.figure()
    plt.plot(freq, response)
    plt.plot(lorentz_freq, lorentz_resp)
    plt.show()
    '''
    res_f = freq[idx]

    bkg = (response[0]+response[-1])/2
    bkg_slp = (response[-1]-response[0])/(freq[-1]-freq[0])
    skw = 0

    mintrans = bkg-response.min()

    Q = 1e4

    popt, pcov = curve_fit(skewed_lorentzian,lorentz_freq,lorentz_resp,p0=[bkg,bkg_slp,skw,mintrans,res_f,Q])

    res = lorentz_resp - skewed_lorentzian(lorentz_freq, *popt)
    '''
    plt.subplot(211)
    plt.plot(freq, response)
    plt.plot(lorentz_freq, lorentz_resp)
    plt.plot(freq, skewed_lorentzian(freq, *popt))
    plt.subplot(212)
    plt.plot(lorentz_freq, res, 'k')
    plt.show()
    '''

    return popt[4]
    

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

def chan2freq(chan,freqs):
    return chan*(freqs[-1]-freqs[0])/freqs.size+freqs[0]

class autoaligner():
    pass
    #def __init__(self, search_order='fwd', )
        # describe all the knobs to turn

def get_fundamental_freqs(responses, freqs, Q_guess=1e4, fit_win=100, plot=False):
    '''
    see docstring for get_fundamental_inds. 
    except here we look at the whole spectrum and fit a lorentzian.
    the return is a 2d array with (f0s, errs), rather than a 1d list.

    This one expects a single resonance in view. Do a rough align before using this.

    TODO: Implement fit_win scale with zoom
    '''
    
    N = responses.shape[0]
    f_points = responses.shape[1]
    results = np.zeros((N, 2))

    # works for 10.5 MHz span
    fit_win = int(np.round(fit_win * f_points/801)) # n points it was tuned on

    if plot:  
        plot_side_len = np.ceil(np.sqrt(N))
        plt.figure()

    for n in range(N):

        bkg = (responses[n][0]+responses[n][-1])/2
        bkg_slp = (responses[n][-1]-responses[n][0])/(freqs[-1]-freqs[0])
        skw = 0

        mintrans = bkg-responses[n].min()
        min_ind = responses[n].argmin()
        res_f = freqs[min_ind]

        win_freq = freqs[max(min_ind - fit_win, 0) : min(min_ind + fit_win, len(responses[n]))]
        win_resp = responses[n][max(min_ind - fit_win, 0) : min(min_ind + fit_win, len(responses[n]))]

        Q = Q_guess

        popt, pcov = curve_fit(skewed_lorentzian,win_freq,win_resp,p0=[bkg,bkg_slp,skw,mintrans,res_f,Q])

        results[n][0] = popt[4]
        results[n][1] = np.sqrt(pcov[4][4])

        if plot:
            plt.subplot(plot_side_len, plot_side_len, n+1)
            plt.plot(freqs, responses[n], 'k.')
            x = np.linspace(win_freq[0], win_freq[-1])
            plt.plot(x, skewed_lorentzian(x, *popt), 'r')
            plt.axis('off')

    if plot:
        plt.show()

    return results

def get_turning_point_fits(responses, coord, coord_poss, start_pos, freqs, fit_win=100, plot=False):

    coord_num = np.where(np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW']) == coord)[0]

    results = get_fundamental_freqs(responses,freqs, fit_win=fit_win, plot=False)
    ffreqs = results[:,0].T
    errs = results[:,1].T
    
    coord_poss = coord_poss.T[0] # quirk of how things are ordred
    print(coord_poss, coord_poss.shape)
    print(ffreqs, ffreqs.shape)
    print(errs, errs.shape)
    print(1/errs, (1/errs).shape)

    if coord == 'dX' or coord == 'dW':
        fit_deg = 4
    else:
        fit_deg = 2

    p, cov = np.polyfit(coord_poss, ffreqs, w=1/(errs)**2, deg=fit_deg, cov="unscaled") # highest degree first in p
    poly_err = np.array([np.sqrt(cov[i][i]) for i in range(len(p))])

    p_unc = np.array([ufloat(p[i], poly_err[i]) for i in range(len(p))])

    # numerically get turning point and uncertainties
    # in this case it's not necessarily the vertex, just the lowest point

    # make gaussian distributions of each parameter
    samples = 1000000 # how many curves to make
    resolution = 1000000 # how many points to plot to find minimum
    p_samples = np.zeros((fit_deg+1, samples))
    tp_samples = np.zeros(samples)
    for i in range(fit_deg+1):
        p_samples[i] = np.random.normal(loc=p_unc[i].n, scale=p_unc[i].s, size=samples)

    # get the lowest point of each of the produced curves
    numerical_est_x = np.linspace(coord_poss[0], coord_poss[-1], resolution)
    for i in range(samples):
        numerical_est_y = np.polyval(p_samples[:,i], numerical_est_x)
        tp_samples[i] = numerical_est_x[np.argmin(numerical_est_y)]

    # get the std of that distribution
    vertex = ufloat(np.mean(tp_smaples), np.std(tp_samples))


    """
    # analytically get turning point and unceratinties
    if fit_deg == 2:
        vertex = -p_unc[1]/(2*p_unc[0])
    elif fit_deg == 4:
        p = -p_unc[1]/(3*p_unc[0])
        q = p**3 + (p_unc[1]*p_unc[2]-3*p_unc[0]*p_unc[3])/(6*p_unc[0]**2)
        r = p_unc[2]/(3*p_unc[0])
        # we just hope we're in the case where there's one vertex
        vertex = (q+sqrt(q**2+(r-p**2)**3))**(1/3) + (q-sqrt(q**2+(r-p**2)**3))**(1/3) + p
    else:
        raise(ValueError,"fit degree has to be 2 or 4")
    """

    print(f"error on vertex: {vertex.s}")

    if plot:
        x = np.linspace(coord_poss[0], coord_poss[-1], 1000)
        plt.figure()
        plt.axhspan(vertex.n-vertex.s, vertex.n+vertex.s, color="deepskyblue", alpha=0.5)
        plt.errorbar(ffreqs, coord_poss, fmt='k.', xerr=errs, capsize=2)
        plt.plot(np.polyval(p,x), x, 'b--')
        bar_x = np.linspace(np.min(ffreqs),np.max(ffreqs)+2e4, 2)
        plt.plot(bar_x,vertex.n*np.ones_like(bar_x), 'b')
        plt.plot(bar_x,start_pos[coord_num]*np.ones_like(bar_x), 'k--')
        plt.figure()
        plt.plot(coord_poss, 0*coord_poss, 'k--')
        plt.plot(coord_poss, ffreqs - np.polyval(p, coord_poss), 'k.')

    return vertex.n
    

def get_fundamental_inds(responses,  freqs, search_order='fwd', search_range=175):
    '''
    fit an equation to the fundamental mode on a mode map

    KNOBS TO TURN:

    start_from (either fwd or rev): We look for peaks near the one just found, but 
    we need to start somewhere. So, we find the leftmost (lowest freq) peak on either the
    top or bottom row, working either top down (fwd) or bottom up (rev).
    Usually the resonance is strongest near the top row, so fwd is default.

    search_range: number of indices to look away from the peak on the prev. row when looking for the next peak. Is adjusted for number of frequency points. (I found this number with 6401 points)

    initial_prominence: prominence required to count as a peak on the first row. more stringent to ignore random noise / artifacts
    subsequent prominence: prom. for peaks on rows after first. less stringent as the resonance can get pretty weak and we still want to see its peaks
    max_width: maximum width for potential peaks. Any above this width are not considered peaks. Also adjusted for number of freq. points

    also feel free to tweak the code itself.
    '''
    N = responses.shape[0]
    f_points = responses.shape[1]
    fundamental_inds = np.zeros_like(responses[:,0], dtype=int)
    all_inds = np.zeros_like(responses[:,0], dtype=object) # for diagnostic purposes

    if search_order != 'fwd' and search_order != 'rev':
        print('get_fundamental got a bad arg for search_order, exiting')
        exit(-1)

    last_peak_pos = 0 # looking for lowest freq peak first
    bounds_start = 0
    bounds_end = responses[0].size-1

    fspan = freqs[-1] - freqs[0]
    print(fspan)

    initial_prominence = 0
    subsequent_prominence = 0
    max_width = 1000000 * fspan/350e6 * f_points/6401 # 6401 is the resolution this was tweaked at
    #wlen = 1000000 * fspan/350e6 * f_points/6401 # 350 MHz is the zoom this was tweaked at
    search_range = int(search_range * f_points/6401)
    for i in range(N):
        if search_order == 'rev':
            n = N - 1 - i
        else:
            n = i
        if i == 0:
            prominence = initial_prominence
        else:
            prominence = subsequent_prominence
        peaks, properties = find_peaks(-responses[n][bounds_start:bounds_end], width=[0,max_width],prominence=prominence)#, wlen=wlen)
        
        if i == 0:
            metric = abs(peaks) # want leftmost peak for first row (min peak pos)
        else:
            metric = abs(peaks+bounds_start-last_peak_pos)*properties['widths'] # narrowest peak closest to prev. peak
        
        '''
        if i == 5:
            plt.figure()
            plt.plot(freqs,responses[i])
            plt.plot(freqs[peaks+bounds_start], responses[i][peaks+bounds_start], 'r.')
            plt.show()
        '''
        if len(peaks) == 0:
            fundamental_inds[n] = -1
            # can be smart about where next search range should be later
            # but there is danger in being too smart...
            continue
        peak_num = np.argmin(metric)
        all_inds[n] = peaks + bounds_start
        fundamental_inds[n] = peaks[peak_num] + bounds_start # must correct for search range limits
        last_peak_pos = fundamental_inds[n]
        bounds_start = last_peak_pos - search_range
        bounds_end = last_peak_pos + search_range

    # skip peak if not found
    skipped = np.where(fundamental_inds < 0)
    fundamental_inds = np.delete(fundamental_inds, skipped)
    all_inds = np.delete(all_inds, skipped)
    
    '''
    param_space = np.linspace(start+incr/2,end-incr/2,N) + start_pos[0]
    plt.plot(freqs[fundamental_inds]*1e-9, param_space, 'r.')
    plot_tuning(responses, freqs, start_pos, coord, start, end)
    plt.figure()
    ind = 5
    plt.plot(freqs,responses[ind])
    plt.plot(freqs[fundamental_inds[ind]], responses[ind][fundamental_inds[ind]], 'r.')
    plt.show()
    '''

    print(fundamental_inds)

    return fundamental_inds, skipped

def get_turning_point(responses, coord, start_pos, start, end, incr, search_range,freqs,plot=False):

    coord_num = np.where(np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW']) == coord)[0]

    fundamental_inds, skipped = get_fundamental_inds(responses,freqs,search_order='rev',search_range=search_range)
    N = responses.shape[0]
    x = np.linspace(start+incr/2,end-incr/2,N) + start_pos[coord_num]
    x = np.delete(x, skipped)
    y = freqs[fundamental_inds]*1e-9

    fit_deg = 2

    p = np.polyfit(x, y, deg=fit_deg) # highest degree first in p

    turning_point = -p[1]/(2*p[0])

    if plot:
        plt.figure()
        tuning_plotter.plot_tuning(responses, freqs, start_pos, coord, start, end)
        plt.plot(y, x, 'r.')
        plt.plot(np.polyval(p,x), x, 'b--')
        plt.plot(freqs*1e-9,turning_point*np.ones_like(freqs), 'b')
        plt.plot(freqs*1e-9,start_pos[coord_num]*np.ones_like(freqs), 'k--')

    return turning_point


def test_function(filename):
    '''
    reads ashley's files
    '''
    with open(filename, 'r') as f:
        y = f.readline().strip()
        x = f.readline().strip()
    y = y.split(',')
    resp = y[1:-2]
    resp = [float(r.strip()) for r in resp]

    x = x.split(', ')
    freq = x[1:-1]
    freq = [float(r.strip()) for r in freq]

    # plt.plot(freq, resp)
    # plt.show()

    filtered_resps = fft_cable_ref_filter(np.array(resp), harmon=5, plot=False)
    filtered_resps = fft_cable_ref_filter(filtered_resps, harmon=6, plot=False)
    time1 = str(time.time())
    with open('C:/Users/FTS/source/repos/axion_detector/spectra/right_top_filtered.txt', 'w') as f:
        f.write('Response (dB), ')
        respo = ''
        for i in filtered_resps:
            respo+= str(i)
            respo += ', '
        f.write(respo)
        f.write('\n Frequency (Hz), ')
        frequ = ''
        for j in freq:
            frequ += str(j)
            frequ += ', '
        f.write(frequ)
    plt.figure()
    plt.plot(freq, filtered_resps)
    plt.show()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', '-f', default=None, help="path/to/file")
    
    args = parser.parse_args()

    
    
    test_function(args.filename)
