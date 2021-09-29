# import stuff

# generate instructions: axis, step size, number of steps (each way?)

# loop n times:
#   take data
#   make incremental move
# then return and loop the other way (-incremental move)

# make plot


import time 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import threading

import nidaqmx
import winsound

from hexachamber import HexaChamber
import na_tracer

class AutoScanner():

    def __init__(self, hexachamber, positioner, na_tracer):
        self.hex = hexachamber
        self.pos = positioner
        self.na = na_tracer 
        self.hexstatus ='init'



    def safety_check(self, danger_volts=0.4, channel='ai0', task_number=1, timeout=30):
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
            while((time_elapsed < timeout or timeout < 0) and self.hexstatus == 'scanning'):
                voltage_btw_plate_cavity = task.read()
                if(voltage_btw_plate_cavity < danger_volts):
                    touching = True
                    print("Plate and cavity are touching!")
                    self.hexstatus = 'stop'
                    err, msg = self.hex.abort_all()
                    print(err)
                    print(msg)
                    winsound.Beep(frequency, duration)
                    break 
                time_elapsed = time.time()-time_start
        print('End safety')
        return touching
    
    


    def tuning_scan_safety(self, tuning_sequence, delay=15):
        '''
        '''
        danger_volts = 0.4
        channel = 'ai0'
        taskno = 1
        timeout = None
        safety_thread = threading.Thread(target=self.safety_check, 
                                    args=[danger_volts, channel, taskno, timeout])
        print('Starting scan...')
        self.hexstatus = 'scanning'
        safety_thread.start()
        responses = None
        for i,step in enumerate(tuning_sequence):
            
            if(self.hexstatus == 'stop'):
                break
            time.sleep(delay)
            if(self.hexstatus == 'stop'):
                break
            ## TODO refactor na_tracer so this monstrosity doesn't happen
            response = na_tracer.get_response(self.na)
            if i == 0:
                responses = np.zeros((len(tuning_sequence), len(response)))
            responses[i] = response
            if(self.hexstatus == 'stop'):
                break
            print('Performing move: ' + str(step))
            self.hex.incremental_move(**step)

        return responses   

def generate_single_axis_seq(coord='dX', incr=0.01, start=0, end=0.1):
    '''
    Generates the list of coordinates to move the hexapod 
    TODO: comment more
    '''
    num_moves = int((end-start)/incr)
    seq = [{coord:incr} for i in range(num_moves)]
    seq.insert(0, {coord:start})
    seq.append({coord:-end})
    return seq
 
def tuning_scan(hex, na, tuning_sequence, delay=15):
    '''
    hex: HexaChamber object
    tuning_sequence: list of dicts of step sizes (dX:val,dY:val,dZ,dU,dV,dW), you get it
    '''

    print('Starting scan...')
    for i,step in enumerate(tuning_sequence):
        time.sleep(delay)
        response = na_tracer.get_response(na)
        if i == 0:
            responses = np.zeros((len(tuning_sequence), len(response)))
        responses[i] = response
        hex.incremental_move(**step)

    return responses    

def plot_tuning(responses):

    plt.imshow(responses, interpolation='none', aspect='auto')
    plt.show()

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

    hex = HexaChamber(host=args.hex_ip, username='Administrator', password=args.hex_password)
    na = na_tracer.initialize_device()

    auto = AutoScanner(hex, None, na)
    #safety_check()

    seq = generate_single_axis_seq(incr=0.02, start=-0.1, end=0.1)
    responses = auto.tuning_scan_safety(seq)
    if(responses is not None):
        np.save('test_responses', responses)
        plot_tuning(responses)

    hex.close()

if __name__ == '__main__':
    main()