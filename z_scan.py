import numpy as np
import matplotlib.pyplot as plt

import argparse

from positioner import Positioner
from hexachamber import HexaChamber
from na_tracer import NetworkAnalyzer
import automate
import analyse as ana

Zposs = np.arange(20,90+10,10)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--hex_ip', default='192.168.254.85',
                    help='IP address to connect to the NewportXPS hexapod')
    parser.add_argument('-j', '--pos_ip', default='192.168.254.23',
                    help='IP address to connect to the NewportXPS positioner')
    parser.add_argument('-p', '--hex_password', help='Password to connect to the NewportXPS hexapod')
    parser.add_argument('-q', '--pos_password', help='Password to connect to the NewportXPS positioner' )
    parser.add_argument('-r', '--reinitialize', action='store_true', 
                        help='Whether to reinitialize the xps machines')
    args = parser.parse_args()
    
    print('****************************')

    
    pos = Positioner(host=args.pos_ip, username='Administrator', password=args.pos_password)
    hexa = HexaChamber(host=args.hex_ip, username='Administrator', password=args.hex_password,xps=pos.get_xps())
    na = NetworkAnalyzer()

    webhook = None
    auto = automate.AutoScanner(hexa, pos, na, webhook)

    output = np.zeros((Zposs.size, 5))
    all_resps = np.zeros((Zposs.size, na.get_pna_freq().size))

    for i,Zpos in enumerate(Zposs):

        input("Please zoom out VNA span to see resonator motion.")
        Znow = pos.get_position()
        Zincr = Zpos - Znow
        pos.incremental_move(Zincr)
        actualZ = pos.get_position()
        input("Please adjust view until only fundamental visible.")
        #automate.autoalign_NM(auto, 1e-3, 1e5,  [0.05, 0.1, 0.1, 0.05, 0.05], max_iters=50, fit_win=200, plot=False)

        fit_well = False
        
        while not fit_well:

            freqs = na.get_pna_freq()
            response = na.get_pna_response()

            popt, pcov = ana.get_lorentz_fit(freqs, response, get_cov=True)

            plt.plot(freqs, response, 'k.')
            plt.plot(freqs, ana.skewed_lorentzian(freqs, *popt), 'r--')
            plt.show()

            userin = input("Is the fit good? [yes]/no\n>")
            if userin == "yes":
                fit_well = True
            else:
                input("Readjust VNA settings until you're happy")

        output[i] = [actualZ, popt[4], np.sqrt(pcov[4][4]), popt[5], np.sqrt(pcov[5][5])] # Z, f, f_err, Q, Q-err
        all_resps[i] = response

    np.save("20221005_ZfQ_with_aligns",output)
    np.save("20221005_zscan_all_resps", all_resps)