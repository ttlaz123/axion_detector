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

print(" Current directory: " + str(os.getcwd()))
# import System
# from System import String

sys.path.append(r'.')
sys.path.append(r'lib')
sys.path.append(r'..')
sys.path.append(r'../alicptfts')
sys.path.append(r'../alicptfts/alicptfts')

# import lib.MC2000B_COMMAND_LIB as mc2000b
# import MC2000B_COMMAND_LIB as mc2000b
from mynewportxps.newportxps.XPS_C8_drivers import XPS, XPSException
from mynewportxps.newportxps.ftp_wrapper import SFTPWrapper, FTPWrapper
from mynewportxps.newportxps import NewportXPS

## TODO put this in some sort of enum or something
CHANGE_VERBOSE = 'z'
CHANGE_DEBUG = 'x'
EXIT = 'q'
CHANGE_HEX_INCREMENT = 'c'
CHANGE_POS_INCREMENT = 'b'
CHANGE_POS_VELOCITY = 'v'
HELP = 'h'
class Positioner:
    MIN_VEL = 0.001
    MAX_VEL = 5
    MAX_ACCEL = 80 

    MIN_INCR = 0.001
    MAX_INCR = 10

    UP = '.'
    DOWN = ','
    ZERO = '0'
    def __init__(self, host, group_name='Group3', stage_name='Group3.Pos',
                    username='Administrator', password='xxxxx', 
                    default_velocity=5, default_increment=1):
        '''
        TODO write comments
        '''
        self.newportxps = None
        try:
            self.newportxps = NewportXPS(host=host, username=username, password=password)
        except XPSException:
            print(' ********** Log in failed. Continuing without positioner *********')

        self.group_name = group_name 
        self.stage_name = stage_name 
        
        self.velocity = default_velocity 
        self.increment = default_increment
    
    def get_xps(self):
        '''
        TODO write comments
        '''
        if(self.newportxps is None):
            print('Not connected to NewportXPS')
            return None
        return self.newportxps._xps 

    def reinitialize(self, kill_groups=True):
        '''
        reinitializes the positioner
        TODO write comments
        '''
        print('STATUS: Reinitializing')
        if(kill_groups):
            self.newportxps.kill_group(self.group_name)

        self.newportxps.initialize_group(self.group_name)
        print('STATUS: Initialized group: ' + self.group_name)
        self.newportxps.home_group(self.group_name)
        print('STATUS: Processed home search')
    
    def generate_velocity_set_command(self, velocity):
        '''
        Generates the command
        TODO write comments
        '''
        cmd_name = 'PositionerSGammaVelocityAndAccelerationSet('
        params = ','.join([str(self.stage_name), str(velocity), str(Positioner.MAX_ACCEL)])
        cmd = cmd_name + params + ')'
        return cmd

    def set_velocity(self, velocity):
        '''
        sets velocity of the positioner
        TODO: put all this in a class
        '''
        
        default_velocity = self.velocity
        try:
            default_velocity=float(velocity)
        except ValueError:
            print('Setting velocity failed: ' + str(velocity) + ' is not a valid float')
            return self.velocity
        
        if(default_velocity < Positioner.MIN_VEL or default_velocity > Positioner.MAX_VEL):
            print('Velocity not within acceptable range: ' + str(default_velocity))
            return self.velocity
        cmd = self.generate_velocity_set_command(velocity)
        try:
            self.newportxps._xps.Send(cmd = cmd)
        except AttributeError:
            print('XPS positioner not connected')
        return default_velocity
    
    def set_incr(self, incr):
        '''
        sets increment of the positioner
        TODO: put all this in a class
        '''
        
        default_incr = self.increment
        try:
            default_incr=float(incr)
        except ValueError:
            print('Setting increment failed: ' + str(incr) + ' is not a valid float')
            return self.increment
        
        if(default_incr < Positioner.MIN_INCR or default_incr > Positioner.MAX_INCR):
            print('Increment not within acceptable range: ' + str(default_incr))
            return self.increment
        self.increment = default_incr
        return default_incr

    def arrow_move(self, pressed_key, verbose=True, debug=False):
        '''
        Moves positioner up and down
        '''
        
        
        if(debug):
            print('    Pressed key: ' + str(pressed_key))
        if(pressed_key == Positioner.UP):
            if(verbose):
                print('Moving up...')
            try:
                self.newportxps.move_stage(stage=self.stage_name, value = self.increment, relative=True)
            except XPSException:
                print('Movement too far or positioner not initialized')
        elif(pressed_key == Positioner.DOWN):
            if(verbose):
                print('Moving down...')
            try:
                self.newportxps.move_stage(stage=self.stage_name, value = -self.increment, relative=True)
            except XPSException:
                print('Movement too far or positioner not initialized')

        
        elif(pressed_key == Positioner.ZERO):
            if(verbose):
                print('Zeroing positioner...')
            self.newportxps.move_stage(stage=self.stage_name, value = 0, relative=False)     

        else:
            if(debug):
                print('        INVALID POSITIONER COMMAND: ' + str(pressed_key))
            return -1 
        if(verbose):
            print('Positioner Command done: ' + pressed_key)

        return 0

    def close(self):
        '''
        close sockets
        '''
        try:             # copy from reboot function
            self.newportxps.ftpconn.close()
            self.newportxps._xps.CloseAllOtherSockets(self._sid)
        except AttributeError:
            pass
    
    
    def arrow_command_instructions():
        down_cmd = 'Move Positioner Down: ' + Positioner.DOWN
        up_cmd = 'Move Positioner Up: ' + Positioner.UP
        zero_cmd = 'Zeros all coordinates on Positioner: ' + Positioner.ZERO 

        cmd = '\n'.join([down_cmd, up_cmd, zero_cmd])
        return cmd

class HexaChamber:
    MIN_VEL = 0.001
    MAX_VEL = 5.000

    UP = 'w'
    DOWN = 's'
    LEFT = 'a'
    RIGHT = 'd'
    TFOR = 'i'
    TBACK = 'k'
    TLEFT = 'j'
    TRIGHT = 'l'
    RCCW = 'u'
    RCW = 'o'

    ZERO = '0'

    def __init__(self, host, 
                 username='Administrator', password='xxxxxx', groupname='HEXAPOD',
                 port=5001, timeout=10, extra_triggers=0, xps=None, default_velocity=1):

        """Establish connection with each part.
        
        Parameters
        ----------
        host : string
            IP address of the XPS controller.

        port : int
            Port number of the XPS controller (default is 5001).

        timeout : int
            Receive timeout of the XPS in milliseconds 
            (default is 1000). Note that the send timeout is 
            set to 1000 milliseconds. See the XPS Programming 
            Manual.

        username : string (default is Administrator)

        password : string (default is Administrator)
        """
        socket.setdefaulttimeout(5.0)
        try:
            host = socket.gethostbyname(host)
        except:
            raise ValueError('Could not resolve XPS name %s' % host)
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.timeout = timeout
        self.groupname = groupname
        self.extra_triggers = extra_triggers

        self.velocity = default_velocity

        self.firmware_version = None

        self.ftpconn = None
        self.ftpargs = dict(host=self.host,
                            username=self.username,
                            password=self.password)
        self.sid = None
        if(xps is None):
            self.xps = XPS()
        else:
            self.xps = xps
        self.sid = self.connect()
    
    def initialize(self, kill_groups=True):
        """Initialize the Hexapod
        
        Parameters
        ----------
        GroupName
        kill_groups
        """
        ### sometimes the groups are already initialized
        if(kill_groups):
            self.xps.KillAll(socketId=None)
        self.xps.GroupInitialize(socketId=None, GroupName=self.groupname)
        print('STATUS: Initialized all groups')
        self.xps.GroupHomeSearch(socketId=None, GroupName=self.groupname)
        print('STATUS: Processed home search')



    def check_error(self, err, msg='', with_raise=True):
        if err != 0:
            err = "%d" % err
            desc = self.xps.errorcodes.get(err, 'unknown error')
            print("XPSError: message= %s, error=%s, description=%s" % (msg, err, desc))
            if with_raise:
                raise XPSException("%s %s [Error %s]" % (msg, desc, err))

    def connect(self):
        self.sid = self.xps.TCP_ConnectToServer(self.host,
                                                  self.port, self.timeout)
        print('Connected to socket: ' + str(self.sid))
        try:
            
            err, val = self.xps.Login(self.sid, self.username, self.password)
            passwordError = -106
            if(int(err) == passwordError ):
                raise XPSException('Incorrect Password: ' + str(err))
            
        except:
            raise XPSException('Login failed for %s and password %s' % (self.host, self.password))
        
        err, val = self.xps.FirmwareVersionGet(self.sid)
        self.firmware_version = val
        self.ftphome = ''

        if 'XPS-D' in self.firmware_version:
            err, val = self.xps.Send(self.sid, 'InstallerVersionGet(char *)')
            self.firmware_version = val
            self.ftpconn = SFTPWrapper(**self.ftpargs)
        else:
            self.ftpconn = FTPWrapper(**self.ftpargs)
            if 'XPS-C' in self.firmware_version:
                self.ftphome = '/Admin'
        return self.sid

    def recenter_hexapod(self, GroupName=None, CoordinateSystem=None, 
                            X=0, Y=0, Z=0, U=0, V=0, W=0):
        '''
        comments here
        '''
        cmd = self.HexapodMoveAbsoluteCmd(GroupName, CoordinateSystem, X, Y, Z, U, V, W)
        err, msg = self.xps.Send(socketId=self.sid, cmd =cmd)
        return err, msg

    def HexapodMoveAbsoluteCmd(self, GroupName=None, CoordinateSystem=None, 
                            X=0, Y=0, Z=0, U=0, V=0, W=0):
        '''
        Comments here
        '''
        if(GroupName is None):
            GroupName = self.groupname
        if(CoordinateSystem is None):
            CoordinateSystem = 'Work'
        
        params = ','.join([GroupName, CoordinateSystem, 
                            str(X), str(Y), str(Z), str(U), str(V), str(W)])
        command_name = 'HexapodMoveAbsolute'
        cmd = command_name + '(' + params + ')'
        return cmd

    def HexapodMoveIncrementalCmd(self, GroupName=None, CoordinateSystem=None, 
                            dX=0, dY=0, dZ=0, dU=0, dV=0, dW=0):
        '''
        Comments here
        '''
        if(GroupName is None):
            GroupName = self.groupname
        if(CoordinateSystem is None):
            CoordinateSystem = 'Work'
        
        params = ','.join([GroupName, CoordinateSystem, 
                            str(dX), str(dY), str(dZ), str(dU), str(dV), str(dW)])
        command_name = 'HexapodMoveIncremental'
        cmd = command_name + '(' + params + ')'
        return cmd 
    
    def incremental_move(self, coord_sys=None, dX=0, dY=0, dZ=0, dU=0, dV=0, dW=0, debug=False):
        '''
        performs the actual movement
        '''
        generated_command = self.HexapodMoveIncrementalCmd(CoordinateSystem=coord_sys,
                                    dX=dX, dY=dY, dZ=dZ, dU=dU, dV=dV, dW=dW)
        if(debug):
            print('        Socket: ' + str(self.sid))
        err, msg = self.xps.Send(socketId=self.sid, cmd = generated_command)
        return err, msg

    def set_velocity(self, vel):
        '''
        sets velocity of manual controller
        '''
        
        default_velocity = self.velocity
        try:
            default_velocity=float(vel)
        except ValueError:
            print('Setting velocity failed: ' + str(vel) + ' is not a valid float')
            return 
        
        if(default_velocity < HexaChamber.MIN_VEL or default_velocity > HexaChamber.MAX_VEL):
            print('Velocity not within acceptable range: ' + str(default_velocity))
            return  
        self.velocity = default_velocity

    def arrow_move(self, pressed_key, verbose=True, debug=False):
        '''
        moves the hexapod depending on pressed keys
        TODO: comment, allow people to change the keybindings?
        '''

        if(debug):
            print('    Pressed key: ' + str(pressed_key))
        if(pressed_key == HexaChamber.LEFT):
            if(verbose):
                print('Moving left...')
            self.incremental_move(dY=self.velocity, debug=debug)
        elif(pressed_key == HexaChamber.RIGHT):
            if(verbose):
                print('Moving right...')
            self.incremental_move(dY=-self.velocity, debug=debug)
        elif(pressed_key == HexaChamber.UP):
            if(verbose):
                print('Moving further...')
            self.incremental_move(dX=self.velocity, debug=debug)
        elif(pressed_key == HexaChamber.DOWN):
            if(verbose):
                print('Moving closer...')
            self.incremental_move(dX=-self.velocity, debug=debug)

        elif(pressed_key == HexaChamber.TLEFT):
            if(verbose):
                print('Tilting left...')
            self.incremental_move(dU=-self.velocity, debug=debug)
        elif(pressed_key == HexaChamber.TRIGHT):
            if(verbose):
                print('Tilting right...')
            self.incremental_move(dU=self.velocity, debug=debug)
        elif(pressed_key == HexaChamber.TFOR):
            if(verbose):
                print('Tilting away...')
            self.incremental_move(dV=self.velocity, debug=debug)
        elif(pressed_key == HexaChamber.TBACK):
            if(verbose):
                print('Tilting toward...')
            self.incremental_move(dV=-self.velocity, debug=debug)
        elif(pressed_key == HexaChamber.RCCW):
            if(verbose):
                print('Rotating Counterclockwise...')
            self.incremental_move(dW=self.velocity, debug=debug)
        elif(pressed_key == HexaChamber.RCW):
            if(verbose):
                print('Rotating Clockwise...')
            self.incremental_move(dW=-self.velocity, debug=debug)    
  
        elif(pressed_key == HexaChamber.ZERO):
            if(verbose):
                print('Centering hexapod...')
            self.recenter_hexapod()   

        else:
            if(debug):
                print('        INVALID HEXAPOD COMMAND: ' + str(pressed_key))
            return -1
        if(verbose):
            print('Hexapod Command done: ' + pressed_key)
        
        return 0
    
    
    def close(self):
        """Close the instrument (socket of the XPS)"""
        try:
            self.check_state('close')
        except Exception:
            pass
        try:             # copy from reboot function
            self.ftpconn.close()
            self.xps.CloseAllOtherSockets(self.sid)
        except Exception:
            pass

    def arrow_command_instructions():
        down_cmd = 'Shift Hexapod Closer: ' + HexaChamber.DOWN
        up_cmd = 'Shift Hexapod Further: ' + HexaChamber.UP
        left_cmd = 'Shift Hexapod Left: ' + HexaChamber.LEFT 
        right_cmd = 'Shift Hexapod Right: ' + HexaChamber.RIGHT 
        tfor_cmd = 'Tilt Hexapod Away: ' + HexaChamber.TFOR 
        tback_cmd = 'Tilt Hexapod Toward: ' + HexaChamber.TBACK 
        tleft_cmd = 'Tilt Hexapod Left: ' + HexaChamber.TLEFT 
        tright_cmd = 'Tilt Hexapod Right: ' + HexaChamber.TRIGHT
        rccw_cmd = 'Rotate Hexapod Counter Clockwise: ' + HexaChamber.RCCW 
        rcw_cmd = 'Rotate Hexapod Clockwise: ' + HexaChamber.RCW
        zero_cmd = 'Zeros all coordinates on Hexapod: ' + HexaChamber.ZERO 

        cmd = '\n'.join([down_cmd, up_cmd, left_cmd, right_cmd, 
                        tfor_cmd, tback_cmd, tleft_cmd, tright_cmd, 
                        rccw_cmd, rcw_cmd, zero_cmd])
        return cmd

 
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