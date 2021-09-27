import sys
import os
import argparse
import ntplib
import time
import csv

from concurrent.futures import ThreadPoolExecutor
import asyncio
from scipy.interpolate import interp1d 
print(os.getcwd())
# import System
# from System import String

sys.path.append(r'../lib')
sys.path.append(r'lib')
sys.path.append(r'..')
sys.path.append(r'../alicptfts')
sys.path.append(r'../alicptfts/alicptfts')

import numpy as np
import random
import time 
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import nidaqmx
from nidaqmx.stream_readers import AnalogSingleChannelReader
from nidaqmx.stream_writers import AnalogSingleChannelWriter

# import lib.MC2000B_COMMAND_LIB as mc2000b
# import MC2000B_COMMAND_LIB as mc2000b
#from mynewportxps import NewportXPS
from mynewportxps.newportxps import NewportXPS
from mynewportxps.newportxps.XPS_C8_drivers import XPSException
#from mynewportxps.newportxps.XPS_C8_drivers import XPSException
from mynewportxps.newportxps.newportxps import withConnectedXPS
from alicptfts.alicptfts import AlicptFTS
import posixpath

##### START RASTOR SCANNER CODE

def convert_to_rastor(x_stamps, y_stamps, v_stamps):
    x_times, x_pos = x_stamps
    y_times, y_pos = y_stamps 
    v_times, v_pos = v_stamps 
    interpx = interp1d(x_times, x_pos)
    interpy = interp1d(y_times, y_pos)
    interpv = interp1d(v_times, v_pos)

    max_y = max(y_pos)
    max_x = max(x_pos)
    min_y = min(y_pos)
    min_x = min(x_pos)
    x_range = int(max_x-min_x)+1
    y_range = int(max_y-min_y)+1

    rastor = np.zeros((x_range, y_range))
    max_x_time = max(x_times)
    max_y_time = max(y_times)
    min_x_time = min(x_times)
    min_y_time = min(y_times)
    for t in v_times:
        if(t < min_x_time or t < min_y_time):
            continue
        if(t > max_x_time or t > max_y_time):
            continue
        v = interpv(t)
        x = interpx(t)
        y = interpy(t)
        rastor[int(x-min_x), int(y-min_y)] = v 
    plt.imshow(rastor)
    plt.show()

def nidaqmx_single_read(time_length, time_resolution, channel='ai1', tasknumber=1, filename=None, titles=None):
    times = []
    readings = []
    n = 0
    taskname = 'Dev' + str(tasknumber)
    voltage_channel = '/'.join([taskname, channel])
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(voltage_channel)
        time0 = time.time()
        time_elapsed = time.time()-time0 
        #print(time_elapsed)
        if(filename is not None):
            with open(filename, 'w', newline='') as out:
                csv_out = csv.writer(out)
                csv_out.writerow(titles)
                while(time_elapsed < time_length):
                    if(n%100 == 0):
                        print('Reading voltage sample ' + str(n))
                    time_start = time.time()
                    x = task.read() 
                    time_end = time.time()
                    t=(time_start + time_end)/2
                    times.append(t)
                    readings.append(x)
                    time.sleep(time_resolution)
                    n+=1
                    csv_out.writerow((t, x))
                    time_elapsed = time.time()-time0 
        else:
            while(time_elapsed < time_length):
                    if(n%100 == 0):
                        print('Reading voltage sample ' + str(n))
                    time_start = time.time()
                    x = task.read() 
                    time_end = time.time()
                    times.append((time_start + time_end)/2)
                    readings.append(x)
                    time.sleep(time_resolution)
                    n+=1
                    time_elapsed = time.time()-time0 
    return times, readings

def read_positions(fts, socket, time_length, time_resolution, group_name, filename=None, titles=None):
    times = []
    readings = []
    time0 = time.time()
    time_elapsed = time.time() - time0 
    n = 0
    if(filename is not None):
        print('Writing to file: ' + filename)
        with open(filename, 'w', newline='') as out:
            csv_out = csv.writer(out)
            csv_out.writerow(titles)
            while(time_elapsed < time_length):
                if(n%100 == 0):
                    print('Reading position sample ' + str(n) + ' after ' + str(time_elapsed) + ' of ' + str(time_length) + ' seconds')
                time_start = time.time()
                x = fts.newportxps.get_stage_position(group_name + '.Pos', socket)
                time_end = time.time()
                #print(time_start)
                #print(time_end)
                t = (time_start + time_end)/2
                times.append(t)
                readings.append(float(x))
                
                
                csv_out.writerow((t, x))
                time.sleep(time_resolution)
                n+=1
        
                time_elapsed = time.time()-time0 
    else:
        while(time_elapsed < time_length):
            if(n%100 == 0):
                print('Reading position sample ' + str(n) + ' after ' + str(time_elapsed) + ' of ' + str(time_length) + ' seconds')
            time_start = time.time()
            x = fts.newportxps.get_stage_position(group_name + '.Pos', socket)
            time_end = time.time()
            #print(time_start)
            #print(time_end)
            t = (time_start + time_end)/2
            times.append(t)
            readings.append(float(x))
            
        
            time.sleep(time_resolution)
            n+=1
    
            time_elapsed = time.time()-time0 
        
    
    return times, readings


def plot_readings_timestamps(readings, timestamps,readings2, timestamps2,  channel, channel2,pos, pos_times, pos2, pos_times2):
    utc_list = [datetime.utcfromtimestamp(t) for t in timestamps]
    utc_list2 = [datetime.utcfromtimestamp(t) for t in timestamps2]
    utc_pos = [datetime.utcfromtimestamp(t) for t in pos_times]
    utc_pos2 = [datetime.utcfromtimestamp(t) for t in pos_times2]
    plt.plot(utc_list, [-r for r in readings], label='Voltage (V) in channel ' + str(channel))
    plt.plot(utc_list2, [-r for r in readings2], label='Voltage (V) in channel ' + str(channel2))
    plt.plot(utc_pos, [p/100 for p in pos], label='group1 position/100')
    plt.plot(utc_pos2, [p/100 for p in pos2], label='group2 position/100')
    plt.xlabel('Time (UTC)')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def initialize_fts(password, num_sockets, IP, reinitialize=False):
    print('Initializing FTS')
    fts = AlicptFTS()
    x = fts.initialize(IP,'Administrator', password, kill_groups=reinitialize)
    for i in range(num_sockets):
        fts.newportxps.connect()

    print('Status: Finish initialization')
    fts.status()
    return fts

def move_group(fts, pos, socket):
    for p in pos:
        pos_spec = p.split('.')
        if pos_spec[0] == 'g1':
            fts.newportxps.move_stage('Group1.Pos', int(pos_spec[1]), False, socket=socket)
        elif pos_spec[0] == 'g2':
            fts.newportxps.move_stage('Group2.Pos', int(pos_spec[1]), False, socket=socket)
    return

def get_pos1(fts, socket):
    pos = fts.newportxps.get_stage_position('Group1.Pos', socket=socket)
    return pos

def get_pos2(fts, socket):
    pos = fts.newportxps.get_stage_position('Group2.Pos', socket=socket)
    return pos

def write_seq(min1, max1, min2, max2, step=5):
    # min1, max1, min2, max2 are all int
    seq = ['g2.' + str(min2), 'g1.' + str(min1)]
    for i in range(min2, max2+step, step):
        if i % (2*step) == i % step:
            seq.append('g1.' + str(min1))
        else:
            seq.append('g1.' + str(max1))
        seq.append('g2.' + str(i))
    return seq

def calc_time(velocity, step, g1_min, g1_max, g2_min, g2_max):
    sweep_time = (g1_max-g1_min)/velocity + 0.5
    num_sweeps = (g2_max - g2_min)/step 
    total_sweep_time = sweep_time * num_sweeps 
    sweep_move_time = (g2_max-g2_min)/velocity 
    homing_time = sweep_time * 4

    return total_sweep_time + sweep_move_time + homing_time

'''
def find_shortest_list(full_list):
    shortest = full_list[0]
    for i in full_list:
        if len(i) <= len(shortest):
            shortest = i
    
    return shortest
'''

def save_csv(x_pos, y_pos, volts, xpos_name, ypos_name, volts_name):
    x_fields = ['x_pos times', 'x_pos']
    x_rows = []
    y_fields = ['y_pos times', 'y_pos']
    y_rows = []
    volts_fields = ['nida reading times', 'volts']
    volts_rows = []
    '''
    shortest_list = find_shortest_list(list(x_pos) + list(y_pos) + list(volts))
    for i in range(len(shortest_list) - 1):
        rows.append([list(x_pos)[0][i], list(x_pos)[1][i], list(y_pos)[0][i], list(y_pos)[1][i], list(volts)[0][i], list(volts)[1][i]])
    '''

    for i in range(len(list(x_pos)[0])):
        x_rows.append([list(x_pos)[0][i], list(x_pos)[1][i]])
    
    for i in range(len(list(y_pos)[0])):
        y_rows.append([list(y_pos)[0][i], list(y_pos)[1][i]])

    for i in range(len(list(volts)[0])):
        volts_rows.append([list(volts)[0][i], list(volts)[1][i]])

    
    with open(xpos_name, 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(x_fields)
        csv_out.writerows(x_rows)

    with open(ypos_name, 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(y_fields)
        csv_out.writerows(y_rows)

    with open(volts_name, 'w', newline='') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(volts_fields)
        csv_out.writerows(volts_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--password', help='Password to connect to the NewportXPS')
    parser.add_argument('-r', '--reinitialize', action='store_true', help='Whether to reinitialize the device')
    parser.add_argument('-t', '--prefix', help='Prefix of filenames')
    args = parser.parse_args()
    password = args.password
    IP_address = '192.168.254.254'

    fts = initialize_fts(password=password, num_sockets=4, IP=IP_address, reinitialize=args.reinitialize)
    
    g1_min = -130
    g1_max = 130
    g2_min = -130
    g2_max = 145
    step= 5
    seq = write_seq(g1_min, g1_max, g2_min, g2_max, step)
    #seq = ['g2.-80', 'g1.-145', 'g1.145', 'g2.-40', 'g1.-145', 'g2.0', 'g1.145', 'g2.60', 'g1.-145', 'g2.120', 'g1.145', 'g2.-30', 'g1.0']
    print(seq)
    
    velocity = 100
    max_accl = 500
    print('Changing velocity')
    fts.newportxps.set_velocity('Group1.Pos', velo=velocity, accl=max_accl)
    fts.newportxps.set_velocity('Group2.Pos', velo=velocity, accl=max_accl)
    print('Velocity set')
    time_length = calc_time(velocity, step, g1_min, g1_max, g2_min, g2_max)
    time_resolution = 0.01
    channel = 'ai1'
    channel2 = 'ai4'

    x_fields = ['x_pos times', 'x_pos']
    y_fields = ['y_pos times', 'y_pos']
    v_fields = ['nida reading times', 'volts']

    if(args.prefix):
        prefix = args.prefix
    else:
        prefix = ''
    xpos_name = prefix + '_x_pos.csv'
    ypos_name = prefix + '_y_pos.csv'
    volts_name = prefix + '_volts.csv'

    with ThreadPoolExecutor(max_workers=4) as executor:
        
        g = executor.submit(move_group, fts, seq, socket=0)
        #a = executor.submit(move_group1, fts,seq1 , socket=0)
        #b = executor.submit(move_group2, fts,seq2 , socket=1)
        e = executor.submit(nidaqmx_single_read, time_length, time_resolution, channel, 1, volts_name, v_fields)
        c = executor.submit(read_positions, fts, 2, time_length, time_resolution, 'Group1', xpos_name, x_fields)
        d = executor.submit(read_positions, fts, 3, time_length, time_resolution, 'Group2', ypos_name, y_fields)
        
    
        nida_times, readings = e.result()
        nida_times2, readings2 = [],[]#f.result()
        pos1_times,pos1  = c.result()
        pos2_times,pos2 = d.result()

    x_pos = (pos1_times, pos1)
    y_pos = (pos2_times, pos2)
    volts = (nida_times, readings)
    #tuples of lists

    if(args.prefix is None):
        prefix = ''
    #save_csv(x_pos, y_pos, volts, args.prefix)

    convert_to_rastor(x_pos, y_pos, volts)
    #plot_readings_timestamps(readings, nida_times,nida_times2, readings2,channel,channel2, pos1, pos1_times, pos2, pos2_times)
    
if __name__ == '__main__':
    main()