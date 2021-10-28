import numpy as np

def fft_cable_ref_filter(responses):

    harmon = 30

    resp_fft = np.fft.rfft(responses, axis=1)

    filted_fft = resp_fft.copy()
    d = 3
    filted_fft[:,harmon-d:harmon+d] = 0
    filted_fft[:,2*harmon] = 0
    filted_resp = np.fft.irfft(filted_fft, n=responses.shape[1])
    
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