import numpy as np
import matplotlib.pyplot as plt
import re
import json
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy import stats
from lmfit import Model

from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

import analyse as ana

# where all the data lives (changes with machine)
# all S11 data, be that mode maps or single spectra
dir_tuning_data = "../data/tuning_data/"
# NM algortihm's best parameter choice at each step
dir_NM_histories = "../data/NM_histories/"
# 2d field maps taken with the disk techinque
dir_field_maps = "../data/field_mapping_data/"
# form factor according to COMSOL integrations
dir_comsol_ints = "../data/form_factor_data/"
# simulated S11 data from which to extract predicted Q
dir_comsol_s11 = "../data/simulated_S11_data/"
# aligned positions of many autoalign attempts compiled into a histogrammable format
dir_align_hists = "../data/autoalign_hist_data/"

def load_align_hist(fname, keep_fails=False):
    """
    load autoalign histogram data.
    parameters: 
     - keep_fails: whether to delete runs where the align failed to converge
       (if true, the fres of the run will be -1 for autoalign failure (probably max iters
       reached) and -2 if it couldn't fit a lorentzian to the aligned spectrum).

    returns:
    init_poss, aligned_poss, aligned_freqs, aligned_freqs_err
    
     - init_poss: (N,6) (N is number of autoaligns performed, succesfully if keep_fails 
       is False) The position of the hexapod when alignment began. Usually set randomly.
       Coords are in usual (X, Y, Z, U, V, W) order.
     - aligned_poss: (N,6) Final positions.
     - aligned_freqs: Resonant frequency at aligned position. Found by lorentz fit. If an
       align fails (usually max_iters reached), this is set to -1. If a fit could not be
       found, this is set to -2.
     - aligned_freqs_err: Fitting error on the resonant frequency. Not the error code!
    """

    fullname = f"{dir_align_hists}/{fname}"
    din = np.load(fullname)

    init_poss = din[:,0:6].reshape(-1,6)
    aligned_poss = din[:,6:12].reshape(-1,6)
    aligned_freqs = din[:,12]
    aligned_freqs_err = din[:,13]

    print(f"loaded {dir_align_hists}/{fname} with {init_poss.shape[0]} aligns")

    # if the run ended early there will be zeros
    end_inds = np.where(aligned_freqs == 0)[0]
    if len(end_inds) > 0:
        end_ind = end_inds[0]
        print(f"Run ended early, after {end_ind} aligns!")
        init_poss = init_poss[:end_ind]
        aligned_poss = aligned_poss[:end_ind]
        aligned_freqs = aligned_freqs[:end_ind]
        aligned_freqs_err = aligned_freqs_err[:end_ind]

    if not keep_fails:
        # if an autoalign ended up somewhere where a resonance couldn't be fit to, or it reported max iters.
        nofit_inds = np.where(aligned_freqs < 0)[0]
        if len(nofit_inds) > 0:
            print(f"Some ({len(nofit_inds)}) aligns failed, here: {nofit_inds}")
        init_poss = np.delete(init_poss, nofit_inds, axis=0)
        aligned_poss = np.delete(aligned_poss, nofit_inds, axis=0)
        aligned_freqs = np.delete(aligned_freqs, nofit_inds)
        aligned_freqs_err = np.delete(aligned_freqs_err, nofit_inds)

    return init_poss, aligned_poss, aligned_freqs, aligned_freqs_err

def load_field_map(fname, return_fres=False):
    """
    load field mapping data taken with the disks. 

    parameters:
     - return_fres: whether to return the fiduical frequency as well as the delta f's.

    returns:
     [fres,] deltas
    
     - fres: [only returned if return_fres] resonant frequency found by fitting with both disks out of the way.
     - deltas: (2,N,N) (N is the number of data points taken in each row/column) 
       the change in frequency from fres at each position in the cavity. Data from the front
       half of the cavity is first (deltas[0]) and the back is second. The data is arranged 
       with (0,0) in top left and axis 0 increasing downwards, axis 1 increasing to the 
       right (so that if you imshow deltas[0] you'll see the cavity map as it is in real
       space). The back half is still viewed from the front, i.e. it is not mirrored. There's
       an option in the plotting function to do that if you like (to e.g. compare with maps
       in COMSOL where you have to look from the back).
    """
    with open(f"{dir_field_maps}/{fname}.csv") as f:
        vals = f.readlines()
        #this works provided the measurements were taken in a square:
        dim = int(np.sqrt((len(vals) - 1) // 2 ))
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

def load_comsol_integrations(fname, colnames=['freq', 'ez', 'e2', 'v']):
    """
    Read the comsol integration csv at fname.
    
    parameters:
     - colnames: key names to give each column when constructing the return dictionary
    
    returns:
    results

     - results: a dict with keys specified by colnames. The first column is saved under 
       colnames[0], etc. The dtype is read as complex, but
       converted to real unless colname for that column is 'ex', 'ey', or 'ez' (electric
       field components).
    
    note:
    This function replaces all instances of 'i' in the target file with 'j' to make numpy 
    happy. ALL.
    """
    fullname = f"{dir_comsol_ints}/{fname}"
    
    print(f"working on file: {fullname}")
    
    # numpy expects j's for complex numbers...
    with open(fullname, 'rt') as f:
        dat = f.read()
        dat = dat.replace('i', 'j')
    with open(fullname, 'wt') as f:
        f.write(dat)

    header = 5  # N of lines to skip at the header
    
    cdat = np.genfromtxt(fullname, skip_header=header, dtype=np.complex_)

    # columns are freq, Ez, E^2, V
    # (integrated over the whole model)

    results = {}
    
    for i, name in enumerate(colnames):
        if name == 'ex' or name == 'ey' or name == 'ez':
            results[name] = cdat[:,i].T
        else:
            results[name] = np.real(cdat[:,i].T)
    
    return results

def load_comsol_s11(fname):
    """
    Loads S11 sweeps simulated by COMSOL

    returns:
    freqs, spec

     - freqs: frequencies at which S11 was simulated
     - spec: S11 response predicted by COMSOL
    """

    fullname = f"{dir_comsol_s11}/{fname}"
    
    header = 8  # N of lines to skip at the header
        
    dat = np.genfromtxt(fullname, skip_header=header)

    return dat[:,0], dat[:,1]

def load_NM_history(fname):
    """
    Loads the optimal position found at each NM iteration.

    returns:
    history

     - history: (N, 6) (N is the number of iterations until the align finished) history[i]
       is the best position NM found at the ith step. Coords are in the usual 
       (X, Y, Z, U, V, W) order.
    """

    fullname = f"{dir_NM_histories}/{fname}"
    history = np.load(fullname)
    # yeah this is a np.load wrapper but needed to document
    return history

def load_spec(fname, return_Z=False, full_path=False):
    """
    Loads a single S11 spectrum whose filename looks like 'YYYY-MM-DD-HH-MM-SS_tag_##Z.npy'.

     - full_path: If false, assumes the data is in ./tuning_data. If true, you get to specify the entire relative or absolute path.
    
    returns:
    [Z,] freqs, spec
    
     - Z: Z position of the positioner when the spectrum was taken. Parsed from filename.
       Should work even for decimals, and for arbitrarily long numbers (uses *shudder* regex)
     - freqs: (freq_npts) Frequency bins when the spectrum was taken.
     - spec: (freq_npts) S11 spectrum from the VNA. Can be complex.
    """

    if not full_path:
        fullname = f"{dir_tuning_data}/{fname}"
    else:
        fullname = fname

    freqs, spec = np.load(fullname)
    freqs = np.real(freqs) # both are saved as complex, don't want that.

    if return_Z:
        numbers = re.split('[^0-9-.]', fname)
        Z = numbers[np.where(numbers)[0][1]]
        retval = (Z, freqs, spec)
    else:
        retval = (freqs, spec)
    return retval

def load_mode_map(fname):
    """
    Load a mode map.

    returns:
    freqs, responses, start_pos, coord, start, end

     - freqs: (freq_npts) Frequency bins of the VNA when this mode map was started 
       (shouldn't change)
     - responses: (N, freq_npts) (N is number of position steps in the map) Each row is an 
       S11 spectrum taken at a given position.
     - start_pos: (6) Position of the hexapod before the mapping began (NOT the location of
       the first row). Usual (X, Y, Z, U, V, W) order. Parsed from filename because
       I didn't know any better.
     - coord: which coordinate is being mapped over
     - start: start position of that coordinate relative to its value in start_pos. 
     - end: end position of that coordinate realtive to its value in start_pos. Use these to
       interpolate the location corresponding to any row in responses.
    """
    
    fullname = f"{dir_tuning_data}/{fname}"
    data = np.load(fullname)
    freqs = data[0]
    responses = data[1:]

    coord_str = fname.split('_')[1]
    numbers = re.split('[^0-9-.]', coord_str)
    start_pos = np.array(numbers[:6], dtype=float)
    start = float(numbers[6])
    end = float(numbers[7])
    coord = fname[-6:-4]

    return freqs, responses, start_pos, coord, start, end

def load_Z_scan(fname):
    """
    Load a Z scan made with automate.pos_z_scan. Different from the other mode maps since
    it needs to talk to the positioner.

    returns:
    freqs, responses, start_pos, Z_poss

     - freqs: (freq_npts) The VNA frequency bins
     - responses: (N, freq_npts) (N here is # of Z positions visited) S11 at each Z position
     - start_pos: (6) starting hexapod position. Usual (X, Y, Z, U, V, W). Shouldn't change.
     - Z_poss: (N) full list of Z positions. Should be 1-1 with rows in responses.
    """

    fullname = f"{dir_tuning_data}/{fname}"

    responses = np.load(f"{fullname}.npy")

    with open(f"{fullname}.json") as f:
        metadata = json.load(f)

    coord_names = ['X', 'Y', 'Z', 'U', 'V', 'W']
    start_pos = [metadata[coord] for coord in coord_names]

    return metadata['freqs'], responses, start_pos, metadata['Z_poss']

def calculate_form_factor(cdat, C_correction_factor=1.483):
    """
    Get form factor at each frequency from comsol intergration data.

    inputs:
     - cdat: comsol integration data dict (loaded using load_comsol_data)
     - C_correction_factor: multiply calculated C by this number before returning. Default val comes from the reduction in volume when the absobing zones are removed.
    
    returns:
    Cs

     - Cs: Form factor at that frequency
    """
    
    Cs = np.abs(cdat['ez'])**2 / (cdat['e2'] * cdat['v']) * C_correction_factor

    return Cs

def calculate_form_factor_distribution(form_factors, displacements, aligned_poss):
    """
    Calculates form factor distributions by interpolating C(X) from simulated data and plugging in aligned position distribution.

    parameters:
     - form_factors (6): aligned form factor, then degraded form factor when wedge moved by displacements[i] in the coords[i] direction.
     - displacements (6): distance in mm or deg. that the wedge was misaligned in the corresponding axis to produce the form factor in form_factors[i]. displacements[0] should be 0.
     - aligned_poss (N, 6): as produced by load_align_hist().
    """
    coords = ["X", "Y", "U", "V", "W"]

    Cs_dist = np.zeros_like(aligned_poss)
    for i in range(len(form_factors)-1):
        positions = aligned_poss[:,i]
        deltas = np.abs(positions - np.mean(positions)) # assumes true position is mean of aligned pos's.
        slope = (form_factors[i+1] - form_factors[0]) / (displacements[i+1] - displacements[0])
        intercept = form_factors[0]

        Cs_dist[:,i] = deltas*slope + intercept

        plt.figure()
        plt.title(coords[i])
        plt.hist(Cs_dist[:,i], color='k', bins=20)

        print(coords[i])
        print(np.mean(Cs_dist[:,i]))
        print(np.std(Cs_dist[:,i]))

    print('-'*30)
    print(np.std(Cs_dist))
    r = 0
    for i in range(5):
        r += np.std(Cs_dist[:,i])**2
    print(np.sqrt(r))
    

def plot_align_hists(aligned_poss, return_stats=False):
    """
    Plot histograms of each coordinate's aligned position, and quote variance (even though not exactly Gaussian).

    parameters:
     - return_stats: If True, returns a (6,3) array of (mean, median, std) for each coord
    """

    retval = None
    if return_stats:
        retval = np.zeros((6,3))

    coords = ["X", "Y", "Z", "U", "V", "W"]

    for i, aligned_pos in enumerate(aligned_poss.T):

        plt.figure()
        plt.title(f"Aligned Position of the Wedge in the {coords[i]} Axis")
        plt.ylabel("Count")
        if coords[i] == "X" or coords[i] == "Y" or coords[i] == "Z":
            scale = 1e3
            plt.xlabel("Position ($\mu$m)")
        else:
            scale = 3.6e3
            plt.xlabel("Position (arcseconds)")
            
        plt.hist((aligned_pos-np.mean(aligned_pos))*scale, bins=15, color='k')

        if return_stats:
            retval[i] = [np.mean(aligned_pos*scale), np.median(aligned_pos*scale), np.std(aligned_pos*scale)]

    return retval

def plot_align_init_corrs(init_poss, aligned_poss):
    """
    Plot the correlations between each aligned position of the wedge in each coord and its initial position as a scatter.
    """

    plt.figure()
    coords = ['X', 'Y', 'Z', 'U', 'V', 'W']
    n = np.arange(init_poss.shape[0])
    #ind = 88
    for i,coord in enumerate(coords):
        plt.subplot(231+i)
        plt.scatter(init_poss[:,i], aligned_poss[:,i], c=n)
        #plt.scatter(init_poss[ind,i], aligned_poss[ind,i], c='red')
        plt.title(f"Alignment Scatter in {coord}")
        plt.ylabel("Aligned Position (hexa coords)")
        plt.xlabel("Initial Position (hexa coords)")

def plot_align_xcorrs(aligned_poss):
    """
    Plot the correlations between each aligned position of the wedge in each coord and each other as a scatter.
    """

    pass

def plot_align_corr_heatmap(init_poss, aligned_poss, skip_z=True):
    """
    Plot a heatmap of each dof's correlation R with each other coord. The diagonal is corr with initial position rather than simply 1.

    parameters:
     - skip_z: Whether to include the Z coordinate in the plot
    """

    textsize = 23
    labelsize = 30

    cmap = "Spectral_r"
    textcolor = 'k'

    if skip_z:
        aligned_poss = np.delete(aligned_poss, 2, axis=1)
        init_poss = np.delete(init_poss, 2, axis=1)
    
    xcorr = np.corrcoef(aligned_poss, rowvar=False)
    
    for i in range(xcorr.shape[1]):
        xcorr[i,i] = np.corrcoef(aligned_poss[:,i], init_poss[:,i], rowvar=False)[0,1]

    f, axs = plt.subplots(xcorr.shape[1], xcorr.shape[1], figsize=(8,8))

    coords = ["X", "Y", "Z", "U", "V", "W"]
    if skip_z:
        del coords[2]

    vmin = 0
    vmax = np.max(np.abs(xcorr))

    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            ax = axs[i][j]
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if i >= j:
                ax.imshow([[np.abs(xcorr[i][j])]], vmin=vmin, vmax=vmax, aspect='auto', cmap=cmap)
                ax.text(0, 0, np.round(xcorr[i][j],3), ha='center', va='center', color=textcolor, fontsize=textsize)
                
                if i == axs.shape[0]-1:
                    ax.set_xlabel(coords[j], fontsize=labelsize)
                if j == 0:
                    ax.set_ylabel(coords[i], fontsize=labelsize)

    plt.subplots_adjust(wspace=0, hspace=0)

    
def plot_Zscan_with_fit(Zscan_fname, S11_fit_fnames, show_fits=True):

    Zfreqs, Zspecs, Zstart_pos, Z_map_poss = load_Z_scan(Zscan_fname)

    # fit S11s
    fit_wins = np.array([700]*3+[400]+[500]+[200])
    results = np.zeros((len(S11_fit_fnames), 5)) # (file, param [Z, fres, dfres, Q, dQ])
    
    for i, fname in enumerate(S11_fit_fnames):
    
        Z, freqs, spec = load_spec(fname, return_Z=True)
    
        peaks, properties = find_peaks(-spec, prominence=0.5)
        win = slice(peaks[0]-fit_wins[i],peaks[0]+fit_wins[i])
        wspec = spec[win]
        wfreqs = freqs[win]
    
        popt, pcov = ana.get_lorentz_fit(wfreqs, wspec, get_cov=True)

        if show_fits:
            plt.figure()
            plt.subplot(211)
            plt.plot(freqs, spec, 'k.')
            plt.plot(wfreqs, ana.skewed_lorentzian(wfreqs, *popt), 'r--')
            plt.subplot(212)
            plt.plot(wfreqs, wspec-ana.skewed_lorentzian(wfreqs, *popt), 'k.')
            plt.show()
    
        results[i] = [Z, popt[4], np.sqrt(pcov[4][4]), popt[5], np.sqrt(pcov[5][5])]


    plt.figure()
    ext = [Zfreqs[0], Zfreqs[-1], Z_map_poss[-1], Z_map_poss[0]]
    plt.imshow(Zspecs, extent=ext, interpolation='none', aspect='auto', cmap='plasma_r')
    plt.plot(results[:,1], results[:,0], 'r.')

    print(results[:,0])
    print(results[0,1]*1e-9,results[-1,1]*1e-9)

def plot_all_Cvsf(all_eigen=True):
    """
    plot the form factor as a function of fundamental frequency as we perturb each misalignment coord.
    
    if all_eigen, use the eigen simulations for all of the form factors. More consistent results, but a bit fewer files (so less positional resolution).
    """

    sim_coords = ['x', 'y', 'v', 'u', 'w']
    hexa_coords = ['X', 'Y', 'U', 'V', 'W']
    if all_eigen:
        distance_arrays = [[5, 10, 30], [30, 60], [6, 12], [3, 6, 9], [3, 6, 9, 12]]
        suffixes = ["_eigen"]*5
    else:
        distance_arrays = [[5, 10, 15, 20, 30], [30, 60], [6, 12], [1.5, 3, 4.5, 6, 7.5, 9], [1.5, 3, 4.5, 6, 7.5, 9]]
        suffixes = ["_wf","_eigen", "_eigen", "_wf", "_wf"]
    units = ["um", "um", "arcmin", "arcmin", "arcmin"]

    linestyles = ['b--', 'g--', 'g', 'b', 'r']

    fontsize = 50
    legendsize = 40
    insetticksize = 35

    f, ax1 = plt.subplots(figsize=(17,15))

    ax2 = plt.axes([0,0,1,1])
    ip = InsetPosition(ax1, [0.11, 0.51, 0.4, 0.4]) # left edge, bot. edge, w, h
    ax2.set_axes_locator(ip)

    mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

    ax1.set_xlabel("$\omega_{010}$ GHz$^{-1}$", fontsize=fontsize)
    ax1.set_ylabel("$C$", fontsize=fontsize)

    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=insetticksize)

    for i in range(len(hexa_coords)):
        fnames = [f'aligned_form_factor{suffixes[i]}.txt'] + [f'd{sim_coords[i]}{distance_arrays[i][j]}{units[i]}_form_factor{suffixes[i]}.txt' for j in range(len(distance_arrays[i]))]

        max_Cs = np.zeros(len(fnames))
        max_C_fs = np.zeros(len(fnames))

        for j, fname in enumerate(fnames):
            cdat = load_comsol_integrations(fname)
            Cs = calculate_form_factor(cdat)
            max_Cs[j] = np.max(Cs)
            max_C_fs[j] = cdat['freq'][np.where(Cs == max(Cs))]

        ax1.plot(max_C_fs,max_Cs, linestyles[i], label=hexa_coords[i], linewidth=5)
        ax2.plot(max_C_fs,max_Cs, linestyles[i], label=hexa_coords[i], linewidth=5)

        pady = 1e-3
        padx = 0.3e-3
        if hexa_coords[i] == "U":
            ax2.set_ylim(min(max_Cs)-pady, max(max_Cs)+pady)
            ax2.set_xlim(min(max_C_fs)-padx, max(max_C_fs)+padx)
        
    ax1.legend(loc='lower right', fontsize=legendsize)
    ax1.grid()
    ax2.grid()
    
def plot_all_CvsX(all_eigen=True):
    """
    plot the form factor as a function of each of the alignment parameters.
    
    if all_eigen, use the eigen simulations for all of the form factors. More consistent results, but a bit fewer files (so less positional resolution).
    """

    sim_coords = ['x', 'y', 'v', 'u', 'w']
    hexa_coords = ['X', 'Y', 'U', 'V', 'W']
    if all_eigen:
        distance_arrays = [[5, 10, 30], [30, 60], [6, 12], [3, 6, 9], [3, 6, 9, 12]]
        suffixes = ["_eigen"]*5
    else:
        distance_arrays = [[5, 10, 15, 20, 30], [30, 60], [6, 12], [1.5, 3, 4.5, 6, 7.5, 9], [1.5, 3, 4.5, 6, 7.5, 9]]
        suffixes = ["_wf","_eigen", "_eigen", "_wf", "_wf"]
    units = ["um", "um", "arcmin", "arcmin", "arcmin"]

    linestyles = ['b--', 'g--', 'g', 'b', 'r']

    f, ax1 = plt.subplots(figsize=(17,15))
    ax2 = ax1.twiny()

    fontsize = 50
    legendsize = 40
    
    ax1.set_xlabel("Position ($\mu$m)", fontsize=fontsize)
    ax2.set_xlabel("Position (arcsec)", fontsize=fontsize)
    ax1.set_ylabel("$C$", fontsize=fontsize)
    
    ax1.tick_params(axis='both', which='major', labelsize=fontsize)
    ax2.tick_params(axis='both', which='major', labelsize=fontsize)

    lines = []

    for i in range(len(hexa_coords)):
        fnames = [f'aligned_form_factor{suffixes[i]}.txt'] + [f'd{sim_coords[i]}{distance_arrays[i][j]}{units[i]}_form_factor{suffixes[i]}.txt' for j in range(len(distance_arrays[i]))]

        max_Cs = np.zeros(len(fnames))

        for j, fname in enumerate(fnames):
            cdat = load_comsol_integrations(fname)
            Cs = calculate_form_factor(cdat)
            max_Cs[j] = np.max(Cs)

        if i < 2:
            plot_ax = ax1
        else:
            plot_ax = ax2
        lines += plot_ax.plot([0]+distance_arrays[i],max_Cs, linestyles[i], linewidth=5, label=hexa_coords[i])

    ax1.legend(lines, [l.get_label() for l in lines], fontsize=legendsize)
    ax1.grid()

def plot_s11(freqs, spec, fit=False, start=0, stop=None, return_params=False, x_axis_index=False):
    """
    Plot an S11, like those read by load_spec or load_comsol_s11.

    parameters:
     - fit: Whether to also perform and plot a fit to the resonator. If magnitude only, fits a skewed Lorentzian fit. If complex, fits a complex Loretzian with cable term taken from an SO repo. 
     - start/stop: The first and last indices that will be used in the fit
     - return_params: Whether to return the popt and pcov from the Lorentzian fit. Returns None if fit==False
     - x_axis_index: If true, the raw index values are used on the x axis, rather than frequency. Nice when looking for correct start/stop

    returns
    """

    f1 = plt.figure()
    ax1 = f1.subplots()

    iscomplex = False
    if spec.dtype == complex:
        iscomplex = True
        mag = np.abs(spec)
        phase = np.unwrap(np.angle(spec))
        f2 = plt.figure()
        ax2 = f2.subplots()
        f3 = plt.figure()
        ax3 = f3.subplots()

    
    winfreqs = freqs[start:stop]
    winspec = spec[start:stop]

    print(freqs)
    print(winfreqs)
    
    if not iscomplex:
        if x_axis_index:
            ax1.plot(spec, 'k.')
        else:
            ax1.plot(freqs, spec, 'k.')
    else:
        if x_axis_index:
            ax1.plot(mag, 'k.')
            ax2.plot(phase, 'k.')
        else:
            ax1.plot(freqs, mag, 'k.')
            ax2.plot(freqs, phase, 'k.')
        ax3.plot(np.real(spec), np.imag(spec), 'k.')

    retval = None
    
    if fit:
        if not iscomplex:
            popt, pcov = ana.get_lorentz_fit(winfreqs, winspec, get_cov=True)
            if return_params:
                retval = popt, pcov
            if x_axis_index:
                x = np.arange(len(freqs))
                winx = x[start:stop]
                ax1.plot(winx,ana.skewed_lorentzian(winfreqs, *popt), 'r--', label="fit")
            else:
                smooth_f = np.linspace(min(winfreqs), max(winfreqs), 1000)
                ax1.plot(smooth_f, ana.skewed_lorentzian(smooth_f, *popt), 'r--', label="fit")
            print(f'fres * GHz^-1: {popt[-2]}+/-{pcov[-2][-2]**(1/2)}')
            print(f'Q: {popt[-1]}+/-{pcov[-1][-1]**(1/2)}')
            plt.legend()

        else:
            retval = full_fit(winfreqs, winspec, restrict_f0=True)
            if x_axis_index:
                x = np.arange(len(freqs))
                winx = x[start:stop]
                ax1.plot(winx, np.abs(retval.eval(f=winfreqs)), 'r--')
                ax2.plot(winx, np.unwrap(np.angle(retval.eval(f=winfreqs))), 'r--')
            else:
                freqs_fine = np.linspace(winfreqs[0], winfreqs[-1],10000)
                ax1.plot(freqs_fine, np.abs(retval.eval(f=freqs_fine)), 'r--')
                ax2.plot(freqs_fine, np.unwrap(np.angle(retval.eval(f=freqs_fine))), 'r--')
            ax3.plot(np.real(retval.eval(f=winfreqs)), np.imag(retval.eval(f=winfreqs)), 'r--')
            print(f'fres * GHz^-1: {retval.params["f_0"].value}')
            print(f'Q: {retval.params["Q_0"].value}')
            print(f'beta: {retval.params["beta"].value}')

    return retval

def align_yaxis(axes):
    """
    Stolen shamelessly from StackOverflow
    """
    y_lims = np.array([ax.get_ylim() for ax in axes])

    # force 0 to appear on all axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize all axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lims = y_new_lims_normalized * y_mags
    for i, ax in enumerate(axes):
        ax.set_ylim(new_lims[i])    
        
def plot_NM_history(history, one_plot=False):
    """
    parameters:
     - one_plot: If true, plot all the coordinates on one plot. Otherwise use separate subplots.
    """

    ticksize = 30
    labelsize = 40
    legendsize = 40

    coords = ["X", "Y", "U", "V", "W"]

    history -= history[:,-1].reshape(5,-1)

    if not one_plot:
    
        plt.figure(figsize=(10,20))
        for i in range(5):
            if i == 0:
                ax1 = plt.subplot(511+i)
                plt.title("Best Position at each NM Step", fontsize=40)
            else:
                plt.subplot(511+i, sharex=ax1)
                plt.subplots_adjust(hspace=0)
            if i != 4:
                plt.tick_params(anxis='x',which='both',bottom=False,top=False,labelbottom=False)
            else:
                plt.xlabel("Nelder-Mead Iteration Number", fontsize=labelsize)
                plt.xticks(fontsize=ticksize)
        
            plt.yticks(fontsize=ticksize)
            if coords[i] == "X" or coords[i] == "Y": # linear units
                plt.plot(history[i]*1e3, 'k')
                plt.ylabel(f"{coords[i]} ($\mu$m)", fontsize=labelsize)
            else: # angular
                plt.plot(history[i]*3600, 'k')
                plt.ylabel(f"{coords[i]} (arcsec)", fontsize=labelsize)

    else:
        linestyles = ['b--', 'g--', 'g', 'b', 'r']
        
        f, ax1 = plt.subplots(figsize=(17,15))
        ax2 = ax1.twinx()
        
        ax1.set_ylabel("Position ($\mu$m)", fontsize=labelsize)
        ax2.set_ylabel("Position (arcsec)", fontsize=labelsize)
        ax1.set_xlabel("Nelder-Mead Iteration", fontsize=labelsize)
        
        ax1.tick_params(axis='both', which='major', labelsize=ticksize)
        ax2.tick_params(axis='both', which='major', labelsize=ticksize)

        lines = []
        for i in range(5):

            if coords[i] == "X" or coords[i] == "Y": # linear units
                lines += ax1.plot(history[i]*1e3, linestyles[i], label=coords[i], linewidth=5)
            else:
                lines += ax2.plot(history[i]*3600, linestyles[i], label=coords[i], linewidth=5)

        align_yaxis((ax1, ax2))
        ax1.legend(lines, [l.get_label() for l in lines], fontsize=legendsize, loc=4)
            
def plot_field_map(deltas, plot_E=True, mirror_rear=True, readjust_for_negatives=False):
    """
    plots field mapping data obtained from the disk perturbation technique
    & loaded by load_field_map

    parameters:
     - show_E: plot E = -sqrt(deltas) instead of raw deltas
     - mirror_rear: flip the rear map along the Y=0 axis (as if you were looking at it from behind, rather than looking through the front half from the front). Useful for comparing to COMSOL screenshots.
     - readjust_for_negatives: if plot_E == True, negative deltas will result in nans. In principle this shouldn't happen, but sometimes the disk was in a spot with more field than elsewhere for the fiduical measurement. This adds a constant to all deltas to make the smallest one (in magnitude) zero. ONLY APPLIES if plot_E == True.
    """

    if plot_E:
        if readjust_for_negatives and np.max(deltas) > 0:
            deltas -= np.max(deltas)
        maps = np.sqrt(-1*deltas)
    else:
        maps = deltas

    if mirror_rear:
        maps[1] = np.flip(maps[1], axis=1)

    titles = ["Field Map Using Resonance Perturbation $--$ Front",
              "Field Map Using Resonance Perturbation $--$ Rear"]
    
    for i in range(2):
        plt.figure()
        plt.title(titles[i])
        plt.imshow(maps[i], aspect='auto', interpolation='none', cmap='Spectral_r')
        plt.colorbar(label='$\\lvert\\textbf{E}(\\textbf{x})\\rvert$ (Arbitrary Units)$')

def plot_mode_map(responses,freqs, start_pos, coord, start, end, cmap='plasma_r'):
    """
    Makes a plot from what load_mode_map gives you.

    parameters:
     - cmap: The cmap to use for the mode map
    """

    coords = np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW'])
    init_param = start_pos[np.where(coords==coord)][0]

    titlesize = 50
    labelsize = 40
    ticksize = 30

    freqs = freqs/10**9 # GHz
    plt.imshow(responses, extent=[freqs[0], freqs[-1], (end+init_param)*1e3, (start+init_param)*1e3], interpolation='none', aspect='auto', cmap=cmap)
    #plt.imshow(responses, interpolation='none', aspect='auto', cmap='plasma_r')
    #plt.title(f"Mode Map for {coord[-1]}", fontsize=titlesize, y=1.01)
    plt.xlabel('$f$ (GHz)', fontsize=labelsize)
    plt.ylabel(f'Wedge {coord[-1]} Position ($\mu$m)', fontsize=labelsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    cb = plt.colorbar()
    cb.set_label("S11 (dB)", fontsize=labelsize)
    cb.ax.tick_params(labelsize=ticksize)

def plot_fres_vs_X(sim_fnames, wins=np.array([[0]*6, [-1]*6]).T, show_fits=False, color='g', excluding_aligned=False, fshift=0, symmetrize=True):
    """
    Plot misalignment position of the wedge vs. fres's found by fitting to the s11 in sim_fnames.
    Call this after plot_mode_map to overplot them seamlessly!

    params:
     - map_fname: name of mode map data file. The kind with the big long ugly name with all the positions in the filename
     - sim_fnames: list of names of simulated s11's. The first fname should be the aligned position. Should have {number}d{coord} in the other filenames to let this know the position of the wedge.
     - wins (sim_fnames.size, 2): A list of start and stop windows for the fitting. By default the full span is used.
     - show_fits: Whether to plot the fits. Good for checking the fits. Must be False to overplot well with plot_mode_map of course.
     - color: color of data points in the final plot (as you would pass to plt.scatter)
     - excluding_aligned: Whether the first file is the aligned position file or not. If True, the function will look for a position in the filename of even the first file.
     - fshift: amount to add to the frequency of each point in the final plot. Corrects for construction error when overplotting with plot_mode_map
     - symmetrize: Whether to plot minus positions as well (no new info since cavity is symmetric)
    """

    fress = np.zeros(len(sim_fnames))
    freserrs = np.zeros(len(sim_fnames))
    positions = np.zeros(len(sim_fnames))
    for i, fname in enumerate(sim_fnames):
        print(f"Working on {fname}")
        if i == 0 and not excluding_aligned:
            positions[i] = 0
        else:
            pos_string = re.search(f"d[xyuvw]\\d+", fname)
            positions[i] = float(pos_string.group()[2:])
        
        wide_simfreqs, wide_spec = load_comsol_s11(fname)
        simfreqs = wide_simfreqs[wins[i,0]:wins[i,1]]
        spec = wide_spec[wins[i,0]:wins[i,1]]
        
        popt, pcov = ana.get_lorentz_fit(simfreqs, spec, get_cov=True)

        fress[i] = popt[-2]
        freserrs[i] = np.sqrt(pcov[-2][-2])

        if show_fits:
            xs = np.arange(wide_simfreqs.size)
            win_xs = xs[wins[i,0]:wins[i,1]]
            smooth_xs = np.linspace(min(win_xs), max(win_xs), 1000)
            smooth_fs = np.linspace(min(simfreqs), max(simfreqs), 1000)
            
            plt.figure()
            plt.plot(xs, wide_spec, 'k.')
            plt.plot(smooth_xs, ana.skewed_lorentzian(smooth_fs, *popt), 'r--', label="fit")
    if show_fits:
        plt.show() # ensure you're fitting well
    
    plt.scatter(fress+fshift, positions, color=color, s=15, linewidths=5)
    if symmetrize:
        plt.scatter(fress+fshift, -positions, color=color, s=15, linewidths=5)

def plot_first_three_modes_comparison(show_filted=False):
    """
    Plot an experimental mode map with X displacement, then overplot the simulated prediction of the resonant frequencies of the first three modes.
    Data & sims taken at 75.19 mm.

    parameters:
     - show_filted: Whether to plot the fft filted mode map (if True) or the raw mode map.
    """

    plt.figure(figsize=(13,10))

    map_fname = "2022-10-13-18-25-14_3.291211567346X-0.5641939352805Y10.00045313989Z-0.09095016416645U0.5974875796197V0.9704551575534W-0.05i0.05fdX.npy"
    sim_fnames = ["20221104_Al_75z_aligned_S11.txt"]+[f"20221104_Al_75z_dx{d}um_S11.txt" for d in [5, 10, 15, 20, 30]]

    freqs, responses, start_pos, coord, start, end = load_mode_map(map_fname)

    _, harmon = ana.auto_filter(freqs, responses[0], return_harmon=True)

    filted_resp = ana.fft_cable_ref_filter(responses, harmon=harmon)

    middle_spec = filted_resp[len(responses)//2]
    mode_map_aligned_win = slice(1000,1070)
    popt = ana.get_lorentz_fit(freqs[mode_map_aligned_win], middle_spec[mode_map_aligned_win])
    fres = popt[-2]*1e-9

    #plt.plot(freqs, middle_spec, 'k.')
    #smoothf = np.linspace(min(freqs[mode_map_aligned_win]), max(freqs[mode_map_aligned_win]), 1000)
    #plt.plot(smoothf, ana.skewed_lorentzian(smoothf, *popt), 'r--')
    
    start_pos = np.array([0]*6) # map taken when aligned, so centering coords there
    
    disp_fwin = slice(500,1800)
    
    disp = responses[:,disp_fwin]
    if show_filted:
        disp = filted_resp[:,disp_fwin]
    plot_mode_map(disp, freqs[disp_fwin], start_pos, coord, start, end, cmap='YlOrRd')

    fshift = -0.01523311329 # fres fit just above minus fres from sim

    dot_c = 'w'

    fund_wins = np.array([[25, *[10]*4, 0], [50, *[60]*4, 20]]).T
    plot_fres_vs_X(sim_fnames, wins=fund_wins, show_fits=False, color=dot_c, fshift=fshift)

    second_wins = np.array([[70, 70, 75, 80, 90], [85, 90, 95, 100, 105]]).T
    plot_fres_vs_X(sim_fnames[1:], wins=second_wins, show_fits=False, color=dot_c, excluding_aligned=True, fshift=fshift)

    third_wins = np.array([[103, 159, 145, 145, 140, 130], [113, 172, 176, 176, 170, 160]]).T
    plot_fres_vs_X(sim_fnames, wins=third_wins, show_fits=False, color=dot_c, fshift=fshift)

def plot_experimental_Qs():

    S11_fit_fnames = ['2022-10-12-17-50-02_zoomed_24Z.npy', '2022-10-12-14-56-10_zoomed_30Z.npy', '2022-10-11-10-33-07_zoomed_50Z.npy', '2022-10-06-14-25-41_zoomed_70Z.npy', '2022-10-10-15-11-46_zoomed_90Z.npy', '2022-10-12-14-47-41_zoomed_92Z.npy']
   
    starts = [1500, 1250, 200, 4050, 1250, 2850]
    stops = [4500, 4200, 2900, 5500, 2650, 3990]
    
    Qs = []
    Qerrs = []
    for i,fname in enumerate(S11_fit_fnames):
        print(f"Working on {fname}")
        popt, pcov = plot_s11(*load_spec(fname), fit=True, start=starts[i], stop=stops[i], x_axis_index=True, return_params=True)
        Qs += [popt[-1]]
        Qerrs += [np.sqrt(pcov[-1][-1])]


def plot_all_C_dists():
    displacements = [0, 5, 30, 6, 3, 3]
    fname_coords = ['x', 'y', 'v', 'u', 'w']
    fname_units = ['um', 'um', 'arcmin', 'arcmin', 'arcmin']
    form_factors = []
    for i,d in enumerate(displacements):
        if i == 0:
            cdat = load_comsol_integrations("aligned_form_factor_eigen.txt")
        else:
            cdat = load_comsol_integrations(f"d{fname_coords[i-1]}{displacements[i]}{fname_units[i-1]}_form_factor_eigen.txt")
        Cs = calculate_form_factor(cdat)
        form_factors += [np.max(Cs)]
        if i == 0:
            print(np.max(Cs))

    calculate_form_factor_distribution(form_factors, displacements, aligned_poss)

def plot_V_vs_fres():

    labelsize=30

    Vs = [2699.62002, 2680.23519, 2626.48753] # cm^3
    fress = [7.576653, 7.52777, 7.39191] # GHz

    ps = np.polyfit(fress, Vs, deg=1)

    print(ps)

    plt.plot(fress, np.polyval(ps, fress), 'r--', lw=3)
    plt.plot(fress, Vs, 'k.', ms=12)

    plt.ylabel("Volume (cm$^3$)", fontsize=labelsize)
    plt.xlabel("Resonant Frequency (GHz)", fontsize=labelsize)

    plt.xticks(fontsize=labelsize)
    plt.yticks(fontsize=labelsize)

def linear_resonator(f, f_0, Q_0, beta):
    num = (beta - 1 - (2j*Q_0*(f-f_0)/f_0))
    den = (beta + 1 + (2j*Q_0*(f-f_0)/f_0))
    return num/den 

def cable_delay(f, delay, phi, f_min):

    return np.exp(1j * (-2 * np.pi * (f-f_min) * delay + phi))

def general_cable(f, delay, phi, f_min, A_mag, A_slope):

    phase_term = cable_delay(f, delay, phi, f_min)
    magnitude_term = ((f-f_min)*A_slope + 1) * A_mag
    return magnitude_term*phase_term

def resonator_cable(f, f_0, Q_0, beta, delay, phi, f_min, A_mag, A_slope):

    resonator_term = linear_resonator(f, f_0, Q_0, beta)
    cable_term = general_cable(f, delay, phi, f_min, A_mag, A_slope)
    return cable_term*resonator_term

def prototype_resonator_cable(f, f_0, Q_0, beta, delay, phi, f_min, A_mag, A_slope, B, C, D, E, F):
    """
    Adapted from the SO resonator cable fitting function, but trying to account for the odd
    periodic structure we see in the prototype single-wedge reflection data by adding a 
    linear fit to the magnitude and a quadratic to the phase, fitting only on the ends
    """

    res_cab_term = resonator_cable(f, f_0, Q_0, beta, delay, phi, f_min, A_mag, A_slope)
    rc_A = np.abs(res_cab_term)
    rc_ph = np.unwrap(np.angle(res_cab_term))

    rc_A += B + C*(f-f_min)
    rc_phase += D + E*(f-f_min) + F*(f-f_min)**2

    return rc_A * np.exp(1j*rc_phase)

def full_fit(freqs, s11, restrict_f0=False):
    """
    Fits a full complex resonator model to get everything we want. 
    Based on https://github.com/simonsobs/sodetlib/blob/master/scripts/resonator_model.py
    which seems to expect s21 measurements, so I fit equation (13) in https://arxiv.org/pdf/2010.06183.pdf
    That function has 1 fewer parameter than the other one, though...
    I'll figure out how they're equivalent later

    Fitting functions defined above
    """

    argmin_s11 = np.abs(s11).argmin()
    fmin = np.min(freqs)
    fmax = np.max(freqs)
    f_0_guess = freqs[argmin_s11]
    Q_min = 0.1 * (f_0_guess / (fmax - fmin))
    delta_f = np.diff(freqs)
    min_delta_f = delta_f[delta_f > 0].min()
    Q_max = f_0_guess / min_delta_f
    Q_guess = np.sqrt(Q_min * Q_max)
    Q_guess = 5e3
    beta_guess = 0.5 # I have no idea
    A_slope, A_offset = np.polyfit(freqs - fmin, np.abs(s11), 1)
    A_mag = A_offset
    #A_mag = -0.33
    A_mag_slope = A_slope/A_mag
    #A_mag_slope=3e-8
    phi_slope, phi_offset = np.polyfit(freqs - fmin, np.unwrap(np.angle(s11)), 1)
    delay = -phi_slope/ (2*np.pi)
    
    totalmodel = Model(resonator_cable)
    params = totalmodel.make_params(f0=f_0_guess,
                                    Q_0=Q_guess,
                                    beta=beta_guess,
                                    delay=delay,
                                    phi=phi_offset,
                                    f_min=fmin,
                                    A_mag=A_mag,
                                    A_slope=A_mag_slope)

    f_range = 1e2
    if restrict_f0:
        params['f_0'].set(min=f_0_guess-f_range, max=f_0_guess+f_range)
    else:
        params['f_0'].set(min=fmin, max=fmax)
    Q_min, Q_max = 1e3, 1e4
    params['Q_0'].set(min=Q_min, max=Q_max)
    params['beta'].set(min=0, max=5)
    params['phi'].set(min=phi_offset-np.pi, max=phi_offset+np.pi)
    params['delay'].set(min=0)
    params['A_mag'].set(min=-10, max=10)
    params['f_min'].set(value=fmin, vary=False)

    result = totalmodel.fit(s11, params, f=freqs)

    return result

def func_sc_pow_reflected(f, fo, Q, del_y, C):
    """
    Taken from code for ADMX sidecar
    """

    if del_y>C: return 0 ## Temp fix
    return -(fo/(2*Q))**2*del_y/((f-fo)**2+(fo/(2*Q))**2)+C

def sidecar_fit_reflection(freqs, s11):
    """
    Code adapted from ADMX for the sidecar cavity reflection fit
    """

    s11_mag = np.abs(s11)
    s11_mag2 = s11_mag**2
    s11_phase = np.unwrap(np.angle(s11))

    # guess initial parameters
    
    # f0
    f0_ind = np.argmin(s11_mag2)
    f0_guess = freqs[f0_ind]
    # normalization factor
    filt_pcnt = 0.33
    s11_filtered = stats.trim1(s11_mag2, filt_pcnt, tail='left')
    C_guess = np.median(s11_filtered)
    # depth (positive)
    dy_guess = C_guess - np.min(s11_mag2)
    # Q
    left_s11_mag2 = s11_mag2[:f0_ind]
    ind_fwhm = np.argmin(np.abs(left_s11_mag2 - (C_guess-dy_guess/2)))
    f1 = freqs[ind_fwhm]
    del_f = 2*(f0_guess-f1)
    Q_guess = f0_guess/del_f

    po_guesses = (f0_guess, Q_guess, dy_guess, C_guess)

    # fit power
    P_popt, P_pcov = curve_fit(func_sc_pow_reflected, freqs, s11_mag2, p0=po_guesses)
    # they have sigma=s11_mag basically, which I'm not sure I understand

    f_0_fit, Q_fit, dy_fit, C_fit = P_popt

    # Now we do a bunch of fitting and interpolating of the phase to find beta.
    # "Deconvolving the line path"
    # gamma is equivalent to s11
    g_cav_mag = s11_mag*np.sqrt(1/C_fit)

    interp_phase = interp1d(freqs, s11_phase, kind='cubic')
    inds = np.arange(len(freqs))
    n = 10
    ends_inds = (inds < n) + (inds-len(freqs) > -n)
    f_ends = freqs[ends_inds]
    phase_ends = s11_phase[ends_inds]
    interp_phase_wo_notch = np.poly1d(np.polyfit(f_ends, phase_ends,1))
    # doesn't that cause some problems, since the slopes are the same but offset, so looking at the ends doesn't give you the real background?
    delay_phase = interp_phase_wo_notch(freqs)
    g_cav_phase = interp_phase(freqs) - delay_phase

    # magnitude at resonance
    g_cav_mag_f0 = np.sqrt(func_sc_pow_reflected(f_0_fit, *P_popt) * 1/C_fit)

    # phase at resonance
    g_cav_interp_phase = interp1d(freqs, g_cav_phase, kind='cubic')
    g_cav_phase_f0 = g_cav_interp_phase(f_0_fit)

    ind = np.argmin(abs(freqs-f_0_fit)) # index closest to resonance
    if sum(g_cav_phase[ind:ind+5]) < sum(g_cav_phase[ind-5:ind]):
        sign_phase = 1
    else:
        sign_phase = -1

    beta = (1+sign_phase*g_cav_mag_f0)/(1-sign_phase*g_cav_mag_f0)

    print(f_0_fit, Q_fit, beta)

    return P_popt, beta
        
if __name__=="__main__":

    spec_fname = "2023-02-06-15-49-43_zoomed_NoneZ.npy"

    S11_fit_fnames = ['2022-10-12-17-50-02_zoomed_24Z.npy', '2022-10-12-14-56-10_zoomed_30Z.npy', '2022-10-11-10-33-07_zoomed_50Z.npy', '2022-10-06-14-25-41_zoomed_70Z.npy', '2022-10-10-15-11-46_zoomed_90Z.npy', '2022-10-12-14-47-41_zoomed_92Z.npy']
    Zscan_fname = "20221010_172745_Z_scan/20221010_172745_Z_scan"

    #NM_history_fname = "20221011_102950_NM_history.npy"
    NM_history_fname = "20221011_124231_NM_history.npy"

    map_fname = "2022-10-13-18-25-14_3.291211567346X-0.5641939352805Y10.00045313989Z-0.09095016416645U0.5974875796197V0.9704551575534W-0.05i0.05fdX.npy"
    sim_fnames = ["20221104_Al_75z_aligned_S11.txt"]+[f"20221104_Al_75z_dx{d}um_S11.txt" for d in [5, 10, 15, 20, 30]]

    align_hist_fname = "autoalign_hist_20220919_143431.npy"


    spec = load_spec(spec_fname)

    margin = 1
    freqs = spec[0][margin:-margin]
    s11 = spec[1][margin:-margin]

    plot_s11(freqs, s11, fit=True)
    plt.show()
    exit()
    
    popt, sc_beta = sidecar_fit_reflection(freqs,s11)

    plt.figure()
    plt.plot(freqs, np.abs(s11), 'k.')
    plt.plot(freqs, func_sc_pow_reflected(freqs, *popt), 'r--')
    
    results = full_fit(*spec)
    results.params.pretty_print()

    freqs_fine = np.linspace(freqs[0], freqs[-1],10000)

    f1 = plt.figure()
    ax1 = f1.subplots()
    f2 = plt.figure()
    ax2 = f2.subplots()

    ax1.plot(freqs, np.abs(s11), 'k.')
    ax1.plot(freqs_fine, np.abs(results.eval(f=freqs_fine)), '--')
    
    ax2.plot(freqs, np.unwrap(np.angle(s11)), 'k.')
    ax2.plot(freqs_fine, np.unwrap(np.angle(results.eval(f=freqs_fine))), '--')

    f3 = plt.figure()
    ax3 = f3.subplots()
    f4 = plt.figure()
    ax4 = f4.subplots()
    
    f0_fit = results.params['f_0'].value
    r = 5e4
    for f0 in np.linspace(f0_fit-r, f0_fit+r, 5):
        
        results.params['f_0'].set(min=-np.inf, max=np.inf, value=f0)
        ax3.plot(freqs, np.abs(s11) - np.abs(results.eval(f=freqs)), '.', label=f"f0={f0}")
        ax4.plot(freqs, np.unwrap(np.angle(s11)) - np.unwrap(np.angle(results.eval(f=freqs))), '.', label=f"f0={f0}")
    results.params['f_0'].set(min=-np.inf, max=np.inf, value=f0_fit)
    ax3.plot(freqs, np.abs(s11) - np.abs(results.eval(f=freqs)), '.', label=f"f0={f0}")
    ax4.plot(freqs, np.unwrap(np.angle(s11)) - np.unwrap(np.angle(results.eval(f=freqs))), '.', label=f"f0={f0}")
    ax3.plot(freqs, 0*freqs, 'k--')
    ax4.plot(freqs, 0*freqs, 'k--')
    plt.legend()

    plt.show()
   
    

    exit()
    
    

    #plot_Zscan_with_fit(Zscan_fname, S11_fit_fnames, show_fits=False)
    #plot_NM_history(load_NM_history(NM_history_fname), one_plot=True)
    #plot_all_CvsX()
    #plot_all_Cvsf()
    #plot_field_map(load_field_map('20220831_132445'))

    #plot_s11(*load_comsol_s11('20221118_Al_90z_S11_hires.txt'), fit=True, start=28, stop=-23)
    #plot_s11(*load_spec("2022-10-06-14-25-41_zoomed_70Z.npy"), fit=True, start=4000, stop=-600)
    init_poss, aligned_poss, aligned_freqs, aligned_freqs_err = load_align_hist(align_hist_fname)
    
    #stats = plot_align_hists(aligned_poss, return_stats=True)
    plot_align_corr_heatmap(init_poss, aligned_poss)

    #plot_first_three_modes_comparison()

    #plot_V_vs_fres()
    
    plt.show()

    
