"""
Written by Tom Liu, 2021 Aug 11
Much of this code is imported from newportxps.py, revamped for hexapod chamber use
This was a rush job, so apologies for the lack of commenting
"""
#!/usr/bin/env python

import sys
import os
import argparse
import ntplib
import time
import enum
import socket

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import msvcrt

import threading
from concurrent.futures import ThreadPoolExecutor

import na_tracer

from positioner import Positioner
from hexachamber import HexaChamber

print(" Current directory: " + str(os.getcwd()))
# import System
# from System import String

sys.path.append(r'.')
sys.path.append(r'lib')
sys.path.append(r'..')
sys.path.append(r'../alicptfts')
sys.path.append(r'../alicptfts/alicptfts')


## TODO put this in some sort of enum or something
CHANGE_VERBOSE = 'z'
CHANGE_DEBUG = 'x'
EXIT = 'q'
PRINT = 'p'
CHANGE_HEX_INCREMENT = 'c'
CHANGE_POS_INCREMENT = 'b'
CHANGE_POS_VELOCITY = 'v'
HELP = 'h'
 
def initialize_positioner(password, IP, username='Administrator', reinitialize=False):
    '''
    initializes positioner
    '''
    print('STATUS: Initializing Positioner')
    pos = Positioner(host=IP, username=username, password=password)
    if(reinitialize):
        pos.reinitialize()
    print('STATUS: Finished Initializing Positioner')
    return pos

def initialize_hexapod(password, IP, username='Administrator', xps=None, reinitialize=False):
    """
    does the initiazliing stuff
    xps needed if we want to control both positioner and hexapod on same computer
    TODO: add comments
    """
    print('STATUS: Initializing Hexapod')
    
    hex = HexaChamber(host=IP, username=username, password=password, xps=xps)
    if(reinitialize):
        hex.initialize()
    print('STATUS: Finished Initializing Hexapod')
    return hex


def move_hex_manual(hex, key, verbose, debug):
    try:
        hex.arrow_move(key, verbose, debug)
    except AttributeError:
        print('XPS hexapod not connected')
def move_pos_manual(pos, key, verbose, debug):
    try:
        pos.arrow_move(key, verbose, debug)
    except AttributeError:
        print('XPS positioner not connected')

def generate_instructions():
    print('************************************************************')
    print('****** Instructions for moving Hexapod and Positioner ******')
    print('************************************************************')
    print('General Commands:')
    print('---------------------')
    print(settings_instructions())
    print('---------------------')
    print('Hexapod Commands:')
    print('---------------------')
    print(HexaChamber.arrow_command_instructions())
    print('---------------------')
    print('Positioner Commands:')
    print('---------------------')
    print(Positioner.arrow_command_instructions())
    print('_________________________')
    return 

def settings_instructions():

    help_cmd = 'To see instructions again: ' + HELP
    verb_cmd = 'Toggle verbose (prints information whenever a command is run if True): ' + CHANGE_VERBOSE 
    debug_cmd = 'Toggle debug (prints debugging information about the current command being run if True): ' + CHANGE_DEBUG
    exit_cmd = 'Exit program: ' + EXIT
    hexincr_cmd = 'Change the increment by which the hexapod moves: ' + CHANGE_HEX_INCREMENT
    posincr_cmd = 'Change the increment by which the positioner moves: '  + CHANGE_POS_INCREMENT
    posvel_cmd = 'Change the velocity by which the positioner moves: '  + CHANGE_POS_VELOCITY

    cmd = '\n'.join([help_cmd, verb_cmd,debug_cmd, hexincr_cmd, posincr_cmd, posvel_cmd, exit_cmd])
    return cmd

def move_xps_machines(hex, pos):
    pressed_key = '0'
    hex_thread = threading.Thread(target=move_hex_manual, args=[ hex, pressed_key])
    pos_thread = threading.Thread(target=move_pos_manual, args=[ pos, pressed_key])
    na = na_tracer.initialize_device()
    fig, ax = plt.subplots(1,1)
    verbose = True
    debug = False

    
    while(True):
        pressed_key = msvcrt.getwch()

        if(pressed_key == EXIT):
            if(hex_thread.is_alive()):
                print('    Cannot exit: Hexapod is still running!')
            elif(pos_thread.is_alive()):
                print('    Cannot exit: Positioner is still running!')
            else:
                break
        elif(pressed_key == HELP):
            generate_instructions()
            continue 
        elif(pressed_key == PRINT):
            position = pos.get_position()
            print('Printing Trace')
            fig, ax = na_tracer.print_trace(na, position, fig=fig, ax=ax)
           
            print('Trace saved')

        elif(pressed_key == CHANGE_DEBUG):
            debug = not debug
            print('~~~~ Debug setting: ' + str(debug) + ' ~~~~')
            continue
        elif(pressed_key == CHANGE_VERBOSE):
            verbose = not verbose
            print('~~~~ Verbose setting: ' + str(verbose) + ' ~~~~')
            continue
        elif(pressed_key == CHANGE_HEX_INCREMENT):
            print('Enter hexapod default velocity (' + str(HexaChamber.MIN_VEL) + ' to ' +
                    str(HexaChamber.MAX_VEL) + '), then press enter: ')
            x = input()
            hex.set_velocity(x)
            continue
        elif(pressed_key == CHANGE_POS_INCREMENT):
            print('Enter positioner default increment (' + str(Positioner.MIN_INCR) + ' to ' +
                    str(Positioner.MAX_INCR) + '), then press enter: ')
            x = input()
            increment = pos.set_incr(x)
            continue
        elif(pressed_key == CHANGE_POS_VELOCITY):
            print('Enter positioner default velocity (' + str(Positioner.MIN_VEL) + ' to ' +
                    str(Positioner.MAX_VEL) + '), then press enter: ')
            x = input()
            velocity = pos.set_velocity(x)
            continue

        
        if(hex_thread.is_alive()):
            if(verbose):
                print('    Hexapod still running')
        else:
            hex_thread = threading.Thread(target=move_hex_manual, args=[ hex, pressed_key, verbose, debug])
            hex_thread.start()
            

        if(pos_thread.is_alive()):
            if(verbose):
                print('    Positioner still running')
        else:
            pos_thread = threading.Thread(target=move_pos_manual, args=[ pos, pressed_key, verbose, debug])
            pos_thread.start()
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--hex_ip', default='192.168.254.254',
                    help='IP address to connect to the NewportXPS hexapod')
    parser.add_argument('-j', '--pos_ip', default='192.168.0.254',
                    help='IP address to connect to the NewportXPS positioner')
    parser.add_argument('-p', '--hex_password', help='Password to connect to the NewportXPS hexapod')
    parser.add_argument('-q', '--pos_password', help='Password to connect to the NewportXPS positioner' )
    parser.add_argument('-r', '--reinitialize', action='store_true', 
                        help='Whether to reinitialize the xps machines')
    args = parser.parse_args()
    
    print('****************************')
    password = args.pos_password
    IP = args.pos_ip
    positioner = initialize_positioner(password, IP = IP, reinitialize=args.reinitialize)
    
    print('****************************')
    password = args.hex_password
    IP = args.hex_ip
    hexapod = initialize_hexapod(password, IP = IP, reinitialize=args.reinitialize, xps=positioner.get_xps())
    print('************* Starting Keyboard Control ***************')
    print('Press ' + HELP + ' for help')
    move_xps_machines(hexapod, positioner)
    print('************ Closing Sockets ****************')
    hexapod.close()
    positioner.close()

if __name__ == '__main__':
    #generate_instructions()
    main()