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
    
    def get_position(self):
        '''
        gets position of positioner
        '''
        cmd = 'GroupPositionCurrentGet(Group1,double *)'
        try:
            pos = self.newportxps.get_stage_position(self.stage_name)
        except AttributeError:
            print('XPS positioner not connected')
        return pos

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

    def incremental_move(self, distance):

        try:
            self.newportxps.move_stage(stage=self.stage_name, value = distance, relative=True)
        except XPSException:
            print('Movement too far or positioner not initialized')
            exit(-1)

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
