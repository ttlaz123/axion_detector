import numpy as np
import matplotlib.pyplot as plt

import argparse

import automate, hexachamber, positioner, na_tracer

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
    password = args.pos_password
    IP = args.pos_ip
    if(IP == 'x'):
        pos = None
    else:
        pos = positioner.Positioner(host=args.pos_ip, username='Administrator', password=args.pos_password)#, stage_name="IMS100V")
    hexa = hexachamber.HexaChamber(host=args.hex_ip, username='Administrator', password=args.hex_password)#,xps=pos.get_xps())
    na = na_tracer.NetworkAnalyzer()

    webhook = None
    auto = automate.AutoScanner(hexa, pos, na, webhook)

    harmon = None
    N = 20
    automate.autoalign(auto, ['dX', 'dY', 'dU', 'dV', 'dW'], [0]*5, N=N, coarse_ranges=np.array([0.1,0.3,0.3,0.1,0.1]), fine_ranges=np.array([0.0]*5), skip_coarse=False, plot_coarse=True, save=False, harmon=None)