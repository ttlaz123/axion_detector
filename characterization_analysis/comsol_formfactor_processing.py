import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.interpolate import interp1d

# Script to read form factor data from COMSOL
# Take in a list of filenames, positions, and DoF
# make a csv titled with the DoF, with max form factor and positions associated
# plots the frequency associated w/ max form factor to ensure the frequency does't jump to a new resonance

def read_comsol_integrations(fname, colnames=['freq', 'ez', 'e2', 'v']):
    """
    Read the comsol integration csv at fname
    returns array of freq, ez, e^2, and volume in complex format (only ez has imag part though)
    """
    
    print(f"working on file: {fname}")
    
    # numpy expects j's for complex numbers...
    with open(fname, 'rt') as f:
        dat = f.read()
        dat = dat.replace('i', 'j')
    with open(fname, 'wt') as f:
        f.write(dat)

    header = 5  # N of lines to skip at the header
    
    cdat = np.genfromtxt(fname, skip_header=header, dtype=np.complex_)

    # columns are freq, Ez, E^2, V
    # (integrated over the whole model)

    results = {}
    
    for i, name in enumerate(colnames):
        if name == 'ex' or name == 'ey' or name == 'ez':
            results[name] = cdat[:,i].T
        else:
            results[name] = np.real(cdat[:,i].T)
    
    return results
    

def formfactor_evolution(DoF, data_path, plot=True, plotCvsfres=False):

    C_correction_factor = 1.483

    if DoF == "X" or DoF == 'x':
        distances = [5, 10, 15, 20, 30] # this is for figuring out fnames
        suffix = "_wf"
        unit = "um"
    if DoF == "Y" or DoF == 'y':
        distances = [30, 60]
        suffix = "_eigen"
        unit = "um"
    if DoF == "U" or DoF == 'u':
        distances = [1.5, 3, 4.5, 6, 7.5, 9]
        suffix = "_wf"
        unit = "arcmin"
    if DoF == "V" or DoF == 'v':
        distances = [6, 12]
        suffix = "_eigen"
        unit = "arcmin"
    if DoF == "W" or DoF == 'w':
        distances = [1.5, 3, 4.5, 6, 7.5, 9]
        suffix = '_wf'
        unit = "arcmin"
        
    fnames = [f'{data_path}/aligned_form_factor{suffix}.txt'] + [f'{data_path}/d{DoF.lower()}{distances[i]}{unit}_form_factor{suffix}.txt' for i in range(len(distances))]

    if unit == "um":
        dists = np.array([0] + distances)*1e-3 # in mm
    if unit == "arcmin":
        dists = np.array([0] + distances)/60 # in deg
    
    Cmaxs = np.zeros(len(fnames)) # dimless
    freqs = np.zeros(len(fnames)) # in GHz
    interpN = 1000
    full_Cs = np.zeros((len(fnames), interpN)) # have to interpolate
    
    for i, fname in enumerate(fnames):

        cdat = read_comsol_integrations(fname)
    
        Cs = np.abs(cdat['ez'])**2 / (cdat['e2'] * cdat['v']) * C_correction_factor
    
        f = cdat['freq']
        freq2C = interp1d(f, Cs)
        fnew = np.linspace(np.min(f), np.max(f), interpN)
        interpd_Cs = freq2C(fnew)
        full_Cs[i] = interpd_Cs
        Cmaxs[i] = np.max(Cs)
        freqs[i] = f[np.where(Cs == np.max(Cs))]

        print(freqs[i])
    
        full_Cs[i][np.where(interpd_Cs == np.max(interpd_Cs))] = np.nan

    if plot:
        if unit == "um":
            hexa_unit = "mm"
        if unit == "arcmin":
            hexa_unit = "deg"
        plt.figure()
        plt.title(f"Form Factor Changes with Misalignment, {DoF} Direction")
        plt.ylabel("Form Factor")
        plt.xlabel(f"{DoF} Displacement ({hexa_unit})")
        plt.plot(dists, Cmaxs, 'k')
        plt.figure()
        plt.plot(dists, freqs, 'k')
        plt.title(f"Frequency of Best Form Factor Mode, {DoF} Direction")
        plt.xlabel(f"{DoF} Displacement ({hexa_unit})")
        plt.ylabel("Frequency (GHz)")

    if plotCvsfres:
        plt.figure()
        plt.title(f"Resonant Frequency Change with Form Factor Change, {DoF} Perturbation")
        plt.ylabel("Resonant Frequency $f_{res}/\mathrm{GHz}$")
        plt.xlabel("Form Factor $C$")
        plt.plot(Cmaxs, freqs, 'k')
        

    

if __name__=="__main__":
    data_path = "form_factor_data"
    DoF = 'X'
    formfactor_evolution(DoF, data_path, plot=True, plotCvsfres=True)
    plt.show()
