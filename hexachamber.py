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

# import lib.MC2000B_COMMAND_LIB as mc2000b
# import MC2000B_COMMAND_LIB as mc2000b
from mynewportxps.newportxps.XPS_C8_drivers import XPS, XPSException
from mynewportxps.newportxps.ftp_wrapper import SFTPWrapper, FTPWrapper
from mynewportxps.newportxps import NewportXPS

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

    PZ = '1'
    MZ = '2'

    ZERO = '0'

    def __init__(self, host, 
                 username='Administrator', password='xxxxxx', groupname='HEXAPOD',
                 port=5001, timeout=10, extra_triggers=0, xps=None, default_velocity=0.1):

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
    
    def abort_all(self):
        command_name = 'GroupMoveAbort'
        cmd = command_name + '(HEXAPOD)'
        err, msg = self.xps.Send(socketId = self.sid, cmd=cmd)
        return err, msg

    def get_position(self):
        '''
        get position of the hexapod tool platform in the work coordinates
        '''

        GroupName = self.groupname
        params = ','.join([GroupName, *['double *']*6])
        command_name = 'HexapodPositionCurrentGet'
        cmd = command_name + '(' + params + ')'

        err,msg = self.xps.Send(socketId=self.sid, cmd=cmd)
        position = np.array(msg.split(','), dtype=float)
        return err, position

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

    def HexapodMoveIncrementalCmd(self, GroupName=None, CoordinateSystem='Work', 
                            dX=0, dY=0, dZ=0, dU=0, dV=0, dW=0):
        '''
        Comments here
        '''
        if(GroupName is None):
            GroupName = self.groupname
        
        params = ','.join([GroupName, CoordinateSystem, 
                            str(dX), str(dY), str(dZ), str(dU), str(dV), str(dW)])
        command_name = 'HexapodMoveIncremental'
        cmd = command_name + '(' + params + ')'
        return cmd 
    
    def incremental_move(self, coord_sys='Work', dX=0, dY=0, dZ=0, dU=0, dV=0, dW=0, debug=False):
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
        elif(pressed_key == HexaChamber.PZ):
            if(verbose):
                print('Moving Plus Z...')
            self.incremental_move(dZ=+self.velocity, debug=debug)    

        elif(pressed_key == HexaChamber.MZ):
            if(verbose):
                print('Moving Minus Z...')
            self.incremental_move(dZ=-self.velocity, debug=debug)   

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
        z_cmd = 'Move Hexapod Up: ' + HexaChamber.PZ 
        zm_cmd = 'Move Hexapod Down: ' + HexaChamber.MZ 
        zero_cmd = 'Zeros all coordinates on Hexapod: ' + HexaChamber.ZERO 

        cmd = '\n'.join([down_cmd, up_cmd, left_cmd, right_cmd, 
                        tfor_cmd, tback_cmd, tleft_cmd, tright_cmd, 
                        rccw_cmd, rcw_cmd, z_cmd, zm_cmd, zero_cmd])
        return cmd