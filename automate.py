# import stuff

# generate instructions: axis, step size, number of steps (each way?)

# loop n times:
#   take data
#   make incremental move
# then return and loop the other way (-incremental move)

# make plot


import time 
import argparse

import nidaqmx
import winsound

HEXSTATUS='scanning'


def safety_check(danger_volts=0.4, channel='ai0', task_number=1, timeout=30):
    '''
    Continually measures the potential difference between the cavity
        and the plate. If the potential difference drops below the critical
        voltage, that implies they are touching, and we should stop moving
        the hexapod or move it back to the original position

    Input:
    TODO finish commenting
    TODO have some way to manually stop the readings

    Returns:
        touching - boolean
    
    '''
    
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    taskname = 'Dev' + str(task_number)
    voltage_channel = '/'.join([taskname, channel])
    touching = False
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(voltage_channel)
        if(timeout is None):
            timeout = -1
        time_start = time.time()
        time_elapsed = time.time() - time_start
        curr_voltage = task.read()
        print('Current voltage: ' + str(curr_voltage))
        while(time_elapsed < timeout and timeout > 0 and HEXSTATUS == 'scanning'):
            voltage_btw_plate_cavity = task.read()
            if(voltage_btw_plate_cavity < danger_volts):
                touching = True
                print("Plate and cavity are touching!")
                
                winsound.Beep(frequency, duration)
                break 
            time_elapsed = time.time()-time_start

    return touching

def generate_single_axis_seq(coord='dX', incr=0.01, start=0, end=1):
    '''
    Generates the list of coordinates to move the hexapod 
    TODO: comment more
    '''
    num_moves = int((end-start)/incr)
    seq = [{coord:incr} for i in range(num_moves)]
    seq.insert(0, {coord:start})
    seq.append({coord:-end})
    return seq

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

    safety_check()

if __name__ == '__main__':
    main()