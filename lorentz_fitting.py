import numpy as np
import matplotlib.pyplot as plt

import argparse

from positioner import Positioner
from na_tracer import NetworkAnalyzer

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--pos_ip', default='192.168.254.23',
                help='IP address to connect to the NewportXPS positioner')
parser.add_argument('-q', '--pos_password', help='Password to connect to the NewportXPS positioner' )
args = parser.parse_args()

pos = Positioner(host=args.pos_ip, username='Administrator', password=args.pos_password)
na = NetworkAnalyzer()

print(pos.get_position())