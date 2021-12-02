import os
import pyvisa as pv 
import matplotlib.pyplot as plt
import time

from scipy.optimize import curve_fit
import numpy as np
 

class NetworkAnalyzer:

    def __init__(self, name=None):
        self.name = name
        self.device = self.initialize_device()
        self.choose_channel()

    def initialize_device(self, name=None):
        rm = pv.ResourceManager()
        if(name is not None):
            na_name = name 
        elif(self.name is not None):
            na_name = self.name 
        else:
            resources = rm.list_resources()
            print('Available resources: ' + str(resources))
            na_name = resources[0]
        print('Using name: ' + na_name)
        device = rm.open_resource(na_name)

        return device 

    def send_command(self, cmd_list):
        cmd = ';'.join(cmd_list)
        print('Sending command: ' + cmd)
        try:
            res = self.device.query(cmd)
        except pv.errors.VisaIOError:
            print('Response not received')
            return 

        print('Response Received')
        return res 

    def choose_channel(self, channel_name='CH1_S11_1'):
        self.device.write(f"CALC:PAR:SEL '{channel_name}'")
    
    def print_old_trace(self, position=None, fig=None, ax=None):
        trace_format = 'FORM4'
        write_cmd = 'OUTPFORM'
        str_res = self.send_command([trace_format, write_cmd])
        if(str_res is None):
            return None, None
        response = format_trace4(str_res)

        lim_cmd = 'OUTPLIML'
        str_res = send_command(na, [lim_cmd])
        if(str_res is None):
            return None, None
        freqs = format_trace4(str_res)

        coefficients = fit_skewedLorentzian(freqs,response)
        print(coefficients)
        fig, ax = plot_trace(freqs, response, position,fit=coefficients)#,fig=fig, ax=ax)
        return fig, ax 

    def save_old_trace(self, fname="test_spec"):
        trace_format = 'FORM4'
        write_cmd = 'OUTPFORM'
        str_res = self.send_command([trace_format, write_cmd])
        response = format_trace4(str_res)

        lim_cmd = 'OUTPLIML'
        str_res = self.send_command([lim_cmd])
        freqs = format_trace4(str_res)

        np.save(fname, (freqs,response))

    def get_old_freqs(self):
        lim_cmd = 'OUTPLIML'
        str_res = self.send_command([lim_cmd])
        freqs = format_trace4(str_res)

        return freqs

    def get_old_response(self):
        trace_format = 'FORM4'
        write_cmd = 'OUTPFORM'
        str_res = self.send_command([trace_format, write_cmd])
        response = format_trace4(str_res)

        return response

    def print_pna_trace(self, position=None, fig=None, ax=None):
        responses = self.get_pna_response()
        freqs = self.get_pna_freq()
        coefficients = fit_skewedLorentzian(freqs,responses)
        print(coefficients)
        fig, ax = plot_trace(freqs, responses, position,fit=coefficients)#,fig=fig, ax=ax)
        return fig, ax 


    def get_pna_freq(self):
        str_res = self.send_command(["SENS:FREQ:STAR?;STOP?"])
        if(str_res is None):
            return
        freqs = np.array(str_res.split(';'), dtype=float)
        start = freqs[0]
        end = freqs[1]

        str_res = self.send_command(["SENSe1:SWEep:POIN?"])
        num_points = int(str_res)

        freqs = np.linspace(start, end, num_points)

        return freqs


    def get_pna_response(self):
        str_res = self.send_command(["CALC:DATA? FDATA"])
        return np.array(str_res.split(','), dtype=float)

    def get_pna_complex_response(self):
        str_res = self.send_command(["CALC:DATA? SDATA"])
        raw_numbers = np.array(str_res.split(','), dtype=float)
        complex_numbers = raw_numbers[::2] + raw_numbers[1::2]*1j
        return complex_numbers


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
            #print('Not a float: ' + str(p))
            pass
    return floats

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





def main():
    
    na = NetworkAnalyzer()

    #print(na.send_command(["SENS:FREQ:STAR?;STOP?"]))
    #print(na.device.query('CALC:PAR:CAT?'))
    #print(na.device.write("DISP:WIND1:STATE ON"))
    print(na.device.write("CALC:PAR:SEL 'CH1_S11_1'"))
    plt.plot(na.get_pna_response())
    plt.show()
    return 

if __name__ == '__main__':
    main()