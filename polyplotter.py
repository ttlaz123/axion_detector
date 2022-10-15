import numpy as np
import matplotlib.pyplot as plt
import re

# where all the data lives (changes with machine)
# all S11 data, be that mode maps or single spectra
dir_tuning_data = "/home/tdyson/coding/axion_detector/tuning_data/"
# NM algortihm's best parameter choice at each step
dir_NM_histories = "/home/tdyson/coding/axion_detector/MN_histories/"
# 2d field maps taken with the disk techinque
dir_field_maps = "/home/tdyson/coding/axion_detector/field_mapping_data/"
# form factor according to COMSOL integrations
dir_comsol_ints = "/home/tdyson/coding/axion_detector/form_factor_data/"
# aligned positions of many autoalign attempts compiled into a histogrammable format
dir_align_hists = "/home/tdyson/coding/axion_detector/autoalign_hist_data/"

def load_align_hists(fname, keep_fails=False):
    """
    load autoalign histogram data.
    options: 
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
    with open(f"{dir_field_maps}/{fname}") as f:
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


if __name__=="__main__":
    Z, _, _ = load_spec("2022-10-06-14-25-41_zoomed_70Z.npy", return_Z=True)
    print(Z)
    


