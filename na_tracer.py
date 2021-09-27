import os
import pyvisa as pv 
import matplotlib.pyplot as plt
import time

from scipy.optimize import curve_fit
import numpy as np
 

def skewedLorentzian(x,bkg,bkg_slp,skw,mintrans,res_f,Q):
    term1 = bkg 
    term2 = bkg_slp*(x-res_f)
    numer = (mintrans+skw*(x-res_f))
    denom = (1+4*Q**2*((x-res_f)/res_f)**2)
    term3 = numer/denom
    return term1 + term2 - term3

 

def fit_skewedLorentzian(f,mag):
    if(isinstance(f, list)):
        f=np.array(f)
    if(isinstance(mag, list)):
        mag = np.array(mag)
    if(np.mean(mag) < 0):
        mag = -mag
    # define the initial values
    bkg = (mag[0]+mag[-1])/2
    bkg_slp = (mag[-1]-mag[0])/(f[-1]-f[0])
    skw = 0

    mintrans = bkg-mag.min()
    res_f = f[mag.argmin()]

    Q = 1e4

    low_bounds = [bkg/2,-1e-3,-1,0,f[0],1e2]
    up_bounds = [bkg*2,1e-3,1,30,f[-1],1e5]

    try:
        popt,pcov = curve_fit(skewedLorentzian,f,mag,p0=[bkg,bkg_slp,skw,mintrans,res_f,Q],method='lm')
        
        #if popt[5]<0:
        #    popt,pcov = curve_fit(skewedLorentzian,f,mag,p0=[bkg,bkg_slp,skw,mintrans,res_f,Q],bounds=(low_bounds,up_bounds))
        #    print('Using bounded curve_fit')
        
    except RuntimeError:
        popt=np.zeros((6,))

    return popt

def format_trace4(string_result):
    '''
    expected format: X.XXXXXXEXX, 0.000000E00\n
    '''
    lines = [x.strip() for x in string_result.split('\n')]
    points = [(line.split(',')[0].strip()) for line in lines]
    floats = []
    for p in points:
        try: 
            floats.append(float(p))
        except ValueError:
            print('Not a float: ' + str(p))
    return floats

def send_command(na, cmd_list):
    cmd = ';'.join(cmd_list)
    print('Sending command: ' + cmd)
    res = na.query(cmd)
    print('Response Received')
    return res 
    

def plot_trace(xs, ys, position, fit=None, title='Axion Cavity Resonance Scanner', folder='spectra', fig=None, ax=None):
    if(ax is None):
        fig, ax = plt.subplots(1,1)
    ax.plot(xs, ys, label='Positioner at ' + str(position) + ' mm', linewidth=3)
    if(fit is not None):
        skewed_lorentzian = [-skewedLorentzian(x, fit[0], fit[1], fit[2], fit[3], fit[4], fit[5]) for x in xs]
        ax.plot(xs, skewed_lorentzian, label='Fit at ' + str(position) + ' mm, Q=' + str(abs(fit[5])), linewidth=2)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Response')
    ax.set_title(title)
    ax.legend()
    time0 = time.time()
    fig.savefig(os.path.join(folder, str(time0)+'.png'))
    return fig, ax

def print_trace(na, position=None, fig=None, ax=None):
    trace_format = 'FORM4'
    write_cmd = 'OUTPFORM'
    str_res = send_command(na, [trace_format, write_cmd])
    response = format_trace4(str_res)

    lim_cmd = 'OUTPLIML'
    str_res = send_command(na, [lim_cmd])
    freqs = format_trace4(str_res)

    coefficients = fit_skewedLorentzian(freqs,response)
    print(coefficients)
    fig, ax = plot_trace(freqs, response, position,fit=coefficients)#,fig=fig, ax=ax)
    return fig, ax 

def save_trace(na, fname="test_spec"):
    trace_format = 'FORM4'
    write_cmd = 'OUTPFORM'
    str_res = send_command(na, [trace_format, write_cmd])
    response = format_trace4(str_res)

    lim_cmd = 'OUTPLIML'
    str_res = send_command(na, [lim_cmd])
    freqs = format_trace4(str_res)

    np.save(fname, (freqs,response))

def get_freqs(na):
    lim_cmd = 'OUTPLIML'
    str_res = send_command(na, [lim_cmd])
    freqs = format_trace4(str_res)

    return freqs

def get_response(na):
    trace_format = 'FORM4'
    write_cmd = 'OUTPFORM'
    str_res = send_command(na, [trace_format, write_cmd])
    response = format_trace4(str_res)

    return response

def initialize_device():
    rm = pv.ResourceManager()
    resources = rm.list_resources()
    na_name = resources[0]
    device = rm.open_resource(na_name)
    return device 

def main():
    
    device = initialize_device()
    fig, ax = print_trace(device)
    fig.show()
    return 

if __name__ == '__main__':
    main()