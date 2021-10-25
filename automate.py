# import stuff

# generate instructions: axis, step size, number of steps (each way?)

# loop n times:
#   take data
#   make incremental move
# then return and loop the other way (-incremental move)

# make plot


import time, datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import threading

import nidaqmx
import winsound

from hexachamber import HexaChamber
from positioner import Positioner
import na_tracer

class AutoScanner():

    def __init__(self, hexachamber, positioner, na_tracer):
        self.hex = hexachamber
        self.pos = positioner
        self.na = na_tracer 
        self.hexstatus ='init'



    def safety_check(self, danger_volts=0.1, channel='ai0', task_number=1, timeout=30):
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
                    print("Plate and cavity are touching! (or the power supply is off...)")
                    self.hexstatus = 'stop'
                    err, msg = self.hex.abort_all()
                    print(err)
                    print(msg)
                    winsound.Beep(frequency, duration)
                    break 
                time_elapsed = time.time()-time_start
        print('End safety')
        return touching
    
    


    def tuning_scan_safety(self, tuning_sequence, delay=0.5, safe_check=False):
        '''
        hex: HexaChamber object
        tuning_sequence: list of dicts of step sizes (dX:val,dY:val,dZ,dU,dV,dW), you get it
        '''
        danger_volts = 0.1
        channel = 'ai0'
        taskno = 1
        timeout = None
        safety_thread = threading.Thread(target=self.safety_check, 
                                    args=[danger_volts, channel, taskno, timeout])
        print('Starting scan...')
        self.hexstatus = 'scanning'
        if(safe_check):
            safety_thread.start()
        responses = None

        freqs = self.na.get_pna_freq()
        for i,step in enumerate(tuning_sequence):
            
            print(f'Performing move: {step} ({i+1} of {len(tuning_sequence)})')

            if len(step.keys()) != 1:
                print('only implemented moving one parameter at once so far!')
                exit(-1)

            param_name = list(step.keys())[0]

            if param_name == 'dU' or param_name == 'dV' or param_name == 'dW':
                coord_sys = 'Tool'
            if param_name == 'dX' or param_name == 'dY' or param_name == 'dZ':
                coord_sys = 'Work'
            
            self.hex.incremental_move(**step, coord_sys=coord_sys)

            if(self.hexstatus == 'stop'):
                break
            time.sleep(delay)
            if(self.hexstatus == 'stop'):
                break
            
            if i == len(tuning_sequence)-1:
                # don't take data after re-centering move
                continue

            total_retries = 10
            for attempt in range(total_retries):

                response = self.na.get_pna_response()
                if(response is None):
                    print(f'VNA asleep!, trying again (attempt {attempt+1}/{total_retries})')
                    continue
                else:
                    break

            if i == 0:
                responses = np.zeros((len(tuning_sequence)-1, len(response)))
            responses[i] = response
            if(self.hexstatus == 'stop'):
                break
        self.hexstatus = 'stop'
        return responses, freqs

def generate_single_axis_seq(coord, incr, start, end):
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
    DON'T USE THIS, USE tuning_scan_safety
    hex: HexaChamber object
    tuning_sequence: list of dicts of step sizes (dX:val,dY:val,dZ,dU,dV,dW), you get it
    '''

    print('Starting scan...')
    for i,step in enumerate(tuning_sequence):
        time.sleep(delay)
        response = na.get_pna_response()
        if i == 0:
            responses = np.zeros((len(tuning_sequence), len(response)))
        responses[i] = response
        hex.incremental_move(**step)

    return responses    

def plot_tuning(responses,freqs, start_pos, coord, start, end):

    coords = np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW'])
    init_param = start_pos[np.where(coords==coord)][0]

    freqs = freqs/10**9 # GHz
    plt.imshow(responses, extent=[freqs[0], freqs[-1], end+init_param, start+init_param], interpolation='none', aspect='auto', cmap='plasma_r')
    plt.xlabel('Frequency [GHz]')
    plt.ylabel(f'Tuning Parameter: {coord[-1]}')
    plt.colorbar()

def save_tuning(responses, freqs, start_pos, coord, start, end):
    data_dir = "C:\\Users\\FTS\\source\\repos\\axion_detector\\tuning_data\\"
    now_str = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    fname = f"{now_str}_{start_pos[0]}X{start_pos[1]}Y{start_pos[2]}Z{start_pos[3]}U{start_pos[4]}V{start_pos[5]}W{start}i{end}f{coord}"
    print(f"Saving to: {data_dir}\\{fname}.npy")
    np.save(f"{data_dir}{fname}", np.vstack((freqs,responses)))

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
    #pos = Positioner(host=args.pos_ip, username='Administrator', password=args.pos_password)
    na = na_tracer.NetworkAnalyzer()

    pos.incremental_move(5)
    pos.incremental_move(-5)
    exit()

    auto = AutoScanner(hex, None, na)
    '''
    coords = np.array(['dX', 'dY', 'dU', 'dV', 'dW'])
    starts = np.array([-0.1, -0.2, -0.6, -0.05, -0.05])
    ends = -1*starts
    incrs = 0.05*ends
    '''
    coords = ['dX']
    starts = np.array([-0.04])
    ends = -1*starts
    incrs = 0.005*ends
    

    err,start_pos = hex.get_position()

    print(f"hexapod started at {start_pos}")

    if err != 0:
        print(f'ERROR {err} with hexapod, exiting')
        hex.close()
        exit(err)

    for i in range(len(coords)):
        seq = generate_single_axis_seq(coord=coords[i], incr=incrs[i], start=starts[i], end=ends[i])
        responses, freqs = auto.tuning_scan_safety(seq)
        if(responses is not None):
            plt.figure()
            save_tuning(responses, freqs, start_pos, coords[i], starts[i], ends[i])
            plot_tuning(responses, freqs, start_pos, coords[i], starts[i], ends[i])
            
    #pos.incremental_move(5)
    hex.incremental_move(dV=1)

    err,start_pos = hex.get_position()

    print(f"next cycle hexapod started at {start_pos}")

    if err != 0:
        print(f'ERROR {err} with hexapod, exiting')
        hex.close()
        exit(err)

    for i in range(len(coords)):
        seq = generate_single_axis_seq(coord=coords[i], incr=incrs[i], start=starts[i], end=ends[i])
        responses, freqs = auto.tuning_scan_safety(seq)
        if(responses is not None):
            plt.figure()
            save_tuning(responses, freqs, start_pos, coords[i], starts[i], ends[i])
            plot_tuning(responses, freqs, start_pos, coords[i], starts[i], ends[i])
            

    #pos.incremental_move(5)
    hex.incremental_move(dV=-1)

    plt.show()

    hex.close()

if __name__ == '__main__':
    main()