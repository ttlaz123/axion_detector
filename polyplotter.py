import numpy as np
import matplotlib.pyplot as plt
import re
import json
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

import analyse as ana

# where all the data lives (changes with machine)
# all S11 data, be that mode maps or single spectra
dir_tuning_data = "/home/tdyson/coding/axion_detector/tuning_data/"
# NM algortihm's best parameter choice at each step
dir_NM_histories = "/home/tdyson/coding/axion_detector/NM_histories/"
# 2d field maps taken with the disk techinque
dir_field_maps = "/home/tdyson/coding/axion_detector/field_mapping_data/"
# form factor according to COMSOL integrations
dir_comsol_ints = "/home/tdyson/coding/axion_detector/form_factor_data/"
# simulated S11 data from which to extract predicted Q
dir_comsol_s11 = "/home/tdyson/coding/axion_detector/simulated_S11_data/"
# aligned positions of many autoalign attempts compiled into a histogrammable format
dir_align_hists = "/home/tdyson/coding/axion_detector/autoalign_hist_data/"

def load_align_hists(fname, keep_fails=False):
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

    print(f"loaded {dir_aligned_hists}/{fname} with {init_poss.shape[0]} aligns")

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

def load_spec(fname, return_Z=False):
    """
    Loads a single S11 spectrum whose filename looks like 'YYYY-MM-DD-HH-MM-SS_tag_##Z.npy'.
    
    returns:
    [Z,] freqs, spec
    
     - Z: Z position of the positioner when the spectrum was taken. Parsed from filename.
       Should work even for decimals, and for arbitrarily long numbers (uses *shudder* regex)
     - freqs: (freq_npts) Frequency bins when the spectrum was taken.
     - spec: (freq_npts) S11 spectrum from the VNA. Can be complex.
    """
    
    fullname = f"{dir_tuning_data}/{fname}"

    freqs, spec = np.load(fullname)

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

def plot_s11(freqs, spec, fit=False, start=0, stop=-1):

    winfreqs = freqs[start:stop]
    winspec = spec[start:stop]

    plt.plot(freqs, spec, 'k.')
    
    if fit:
        popt, pcov = ana.get_lorentz_fit(winfreqs, winspec, get_cov=True)
        smooth_f = np.linspace(min(winfreqs), max(winfreqs), 1000)
        plt.plot(smooth_f, ana.skewed_lorentzian(smooth_f, *popt), 'r--', label="fit")
        print(f'fres * Hz^-1: {popt[-2]}+/-{pcov[-2][-2]**(1/2)}')
        print(f'Q: {popt[-1]}+/-{pcov[-1][-1]**(1/2)}')
    
    if fit:
        plt.legend()
    

def plot_NM_history(history):

    ticksize = 20
    labelsize = 30
    
    plt.figure(figsize=(10,20))
    coords = ["X", "Y", "U", "V", "W"]
    for i in range(5):
        if i == 0:
            ax1 = plt.subplot(511+i)
            plt.title("Best Position at eapppch NM Step", fontsize=40)
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
            plt.plot(history[i] - history[i][-1], 'k')
            plt.ylabel(f"{coords[i]} (mm)", fontsize=labelsize)
        else: # angular
            plt.plot((history[i] - history[i][-1])*3600, 'k')
            plt.ylabel(f"{coords[i]} (arcsec)", fontsize=labelsize)

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

def plot_mode_map(responses,freqs, start_pos, coord, start, end):
    """
    Makes a plot from what load_mode_map gives you.
    """

    coords = np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW'])
    init_param = start_pos[np.where(coords==coord)][0]

    freqs = freqs/10**9 # GHz
    plt.imshow(responses, extent=[freqs[0], freqs[-1], (end+init_param)*1e3, (start+init_param)*1e3], interpolation='none', aspect='auto', cmap='plasma_r')
    #plt.imshow(responses, interpolation='none', aspect='auto', cmap='plasma_r')
    plt.title(f"Mode Map for {coord[-1]}", fontsize=30, y=1.01)
    plt.xlabel('Frequency (GHz)', fontsize=20)
    plt.ylabel(f'{coord[-1]} Position (um)', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    cb = plt.colorbar()
    cb.set_label("S11 (dB)", fontsize=20)
    cb.ax.tick_params(labelsize=20)

def plot_fres_vs_X(sim_fnames, wins=np.array([[0]*6, [-1]*6]).T, show_fits=False, fmt='gx', excluding_aligned=False, fshift=0):
    """
    Plot misalignment position of the wedge vs. fres's found by fitting to the s11 in sim_fnames.
    Call this after plot_mode_map to overplot them seamlessly!

    params:
     - map_fname: name of mode map data file. The kind with the big long ugly name with all the positions in the filename
     - sim_fnames: list of names of simulated s11's. The first fname should be the aligned position. Should have {number}d{coord} in the other filenames to let this know the position of the wedge.
     - wins (sim_fnames.size, 2): A list of start and stop windows for the fitting. By default the full span is used.
     - show_fits: Whether to plot the fits. Good for checking the fits. Must be False to overplot well with plot_mode_map of course.
     - fmt: format of data points in the final plot (as you would pass to plt.errorbar)
     - excluding_aligned: Whether the first file is the aligned position file or not. If True, the function will look for a position in the filename of even the first file.
     - fshift: amount to add to the frequency of each point in the final plot. Corrects for construction error when overplotting with plot_mode_map
    """

    print("[0-9]*{coord.lower()}")

    fress = np.zeros(len(sim_fnames))
    freserrs = np.zeros(len(sim_fnames))
    positions = np.zeros(len(sim_fnames))
    for i, fname in enumerate(sim_fnames):
        print(f"Working on {fname}...")
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
            plt.show() # ensure you're fitting well
    
    plt.errorbar(fress+fshift, positions, xerr=freserrs, fmt=fmt, capsize=8, markersize=15)

if __name__=="__main__":

    S11_fit_fnames = ['2022-10-12-17-50-02_zoomed_24Z.npy', '2022-10-12-14-56-10_zoomed_30Z.npy', '2022-10-11-10-33-07_zoomed_50Z.npy', '2022-10-06-14-25-41_zoomed_70Z.npy', '2022-10-10-15-11-46_zoomed_90Z.npy', '2022-10-12-14-47-41_zoomed_92Z.npy']
    Zscan_fname = "20221010_172745_Z_scan/20221010_172745_Z_scan"

    NM_history_fname = "20221011_102950_NM_history.npy"

    map_fname = "2022-10-13-18-25-14_3.291211567346X-0.5641939352805Y10.00045313989Z-0.09095016416645U0.5974875796197V0.9704551575534W-0.05i0.05fdX.npy"
    sim_fnames = ["20221104_Al_75z_aligned_S11.txt"]+[f"20221104_Al_75z_dx{d}um_S11.txt" for d in [5, 10, 15, 20, 30]]

    #plot_Zscan_with_fit(Zscan_fname, S11_fit_fnames, show_fits=False)
    #plot_NM_history(load_NM_history(NM_history_fname))
    #plot_all_CvsX()
    #plot_all_Cvsf()
    #plot_field_map(load_field_map('20220831_132445'))

    #plot_s11(*load_comsol_s11('20221104_Al_70z_S11_hires.txt'), fit=True)
    #plot_s11(*load_spec("2022-10-06-14-25-41_zoomed_70Z.npy"), fit=True, start=4000, stop=-600)

    freqs, responses, start_pos, coord, start, end = load_mode_map(map_fname)

    _, harmon = ana.auto_filter(freqs, responses[0], return_harmon=True)

    filted_resp = ana.fft_cable_ref_filter(responses, harmon=harmon)

    middle_spec = filted_resp[len(responses)//2+1]
    mode_map_aligned_win = slice(1000,1070)
    popt = ana.get_lorentz_fit(freqs[mode_map_aligned_win], middle_spec[mode_map_aligned_win])
    fres = popt[-2]*1e-9

    plt.plot(freqs, middle_spec, 'k.')
    smoothf = np.linspace(min(freqs[mode_map_aligned_win]), max(freqs[mode_map_aligned_win]), 1000)
    plt.plot(smoothf, ana.skewed_lorentzian(smoothf, *popt), 'r--')

    plt.show()
    
    plt.plot(fres, 0, 'o')
    
    start_pos = np.array([0]*6) # map taken when aligned, so centering coords there
    plot_mode_map(filted_resp, freqs, start_pos, coord, start, end)

    fshift = -0.01558898374 # fres fit just above minus fres from sim 

    fund_wins = np.array([[25, *[10]*4, 0], [50, *[60]*4, 20]]).T
    plot_fres_vs_X(sim_fnames, wins=fund_wins, show_fits=False, fmt='gx', fshift=fshift)

    second_wins = np.array([[70, 70, 75, 80, 90], [85, 90, 95, 100, 105]]).T
    plot_fres_vs_X(sim_fnames[1:], wins=second_wins, show_fits=False, fmt='x', excluding_aligned=True, fshift=fshift)
    
    plt.show()

    
