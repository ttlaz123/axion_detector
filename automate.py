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

from scipy.signal import find_peaks
import analyse

import nidaqmx
import winsound

import requests
from discord import Webhook, RequestsWebhookAdapter

from hexachamber import HexaChamber
from positioner import Positioner
import na_tracer

class AutoScanner():

    def __init__(self, hexachamber, positioner, na_tracer, webhook):
        self.hexa = hexachamber
        self.pos = positioner
        self.na = na_tracer 
        self.webhook = webhook
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
        
        frequency = 600  # Set Frequency To 2500 Hertz
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
                    winsound.Beep(frequency, duration)
                    break 
                time_elapsed = time.time()-time_start
        print('End safety')
        return touching

    def incremental_move(self, step):
        """
        step: {"coord": increment}

        move along coord by increment, using hexa unless cood is 'dZ' in which case positioner is used
        """

        print(f'Performing move: {step} ')

        param_name = list(step.keys())[0]

        if param_name == 'dU' or param_name == 'dV' or param_name == 'dW':
            coord_sys = 'Tool'
        if param_name == 'dX' or param_name == 'dY':
            coord_sys = 'Work'
        
        if param_name != 'dZ':
            self.hexa.incremental_move(**step, coord_sys=coord_sys)
        else:
            self.pos.incremental_move(step['dZ'])

    
    def tuning_scan_safety(self, tuning_sequence, delay=0.5, safe_check=True, DATA_TYPE=''):
        '''
        hex: HexaChamber object
        tuning_sequence: list of dicts of step sizes (dX:val,dY:val,dZ,dU,dV,dW), you get it
        '''
        danger_volts = 0.1 # threshold below which contact is detected
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

        time.sleep(delay)
        if(self.hexstatus == 'stop'):
            print("plate and cavity touching at the beginning of the run!! (probably power supply off...)")
            exit(-1)

        freqs = self.na.get_pna_freq()
        for i,step in enumerate(tuning_sequence):
            
            print(f'Iteration {i+1} of {len(tuning_sequence)}')

            if len(step.keys()) != 1:
                print('only implemented moving one parameter at once so far!')
                exit(-1)

            if(self.hexstatus == 'stop'):
                # undo previous step
                reverse_step = {k: -1*v for k, v in tuning_sequence[i-1].items()}
                self.incremental_move(reverse_step)
                break

            self.incremental_move(step)

            time.sleep(delay)
            if(self.hexstatus == 'stop'):
                # undo step done just now
                reverse_step = {k: -1*v for k, v in tuning_sequence[i].items()}
                self.incremental_move(reverse_step)
                break
            
            if i == len(tuning_sequence)-1:
                # don't take data after re-centering move
                continue

            total_retries = 10
            for attempt in range(total_retries):

                response = self.na.get_pna_response()
                if(response is None):
                    print(f'VNA not responding!, trying again (attempt {attempt+1}/{total_retries})')
                    continue
                else:
                    break

            if i == 0:
                responses = np.zeros((len(tuning_sequence)-1, len(response)))
            responses[i] = response
            if(self.hexstatus == 'stop'):
                # undo step done just now
                reverse_step = {k: -1*v for k, v in tuning_sequence[i].items()}
                self.incremental_move(reverse_step)
                break

        collision = False
        if self.hexstatus == 'stop':
            print('scan aborted because of collision!')
            collision = True
        self.hexstatus = 'stop'
        return responses, freqs, collision

def generate_single_axis_seq(coord, incr, start, end):
    '''
    Generates the list of coordinates to move the hexapod 
    TODO: comment more
    '''
    num_moves = int(np.round((end-start)/incr))
    seq = [{coord:incr} for i in range(num_moves)]
    seq.insert(0, {coord:start})
    seq.append({coord:-end})
    return seq
 
def tuning_scan(hexa, na, tuning_sequence, delay=15):
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
        hexa.incremental_move(**step)

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

def scan_one(auto, coord, start, end, incr, plot=True, save=True):
    
    err,start_pos = auto.hexa.get_position()
    if err != 0:
        print(f'ERROR {err} with hexapod, exiting')
        auto.hexa.close()
    
        exit(err)
    start_pos[2] = auto.pos.get_position()

    seq = generate_single_axis_seq(coord=coord, incr=incr, start=start, end=end)
    responses, freqs, collision = auto.tuning_scan_safety(seq, delay=0.2)

    if collision:
        auto.webhook.send(f"COLLISION! Scan of {coord} aborted.")
        exit()

    if plot:
        plt.figure(figsize=[8,6])
        plot_tuning(responses, freqs, start_pos, coord, start, end)
    if save:
        save_tuning(responses, freqs, start_pos, coord, start, end)
    
    auto.webhook.send(f"Scan of {coord} COMPLETE")

    return responses


def scan_many(auto, coords, starts, ends, incrs, plot=True, save=True):

    err,start_pos = auto.hexa.get_position()
    if err != 0:
        print(f'ERROR {err} with hexapod, exiting')
        auto.hexa.close()
        exit(err)

    mode_maps = None # (coord numbr, responses)
    for i in range(len(coords)):
        seq = generate_single_axis_seq(coord=coords[i], incr=incrs[i], start=starts[i], end=ends[i])
        responses, freqs, collision = auto.tuning_scan_safety(seq, delay=0.2)

        if collision:
            auto.webhook.send(f"COLLISION! Scan of {coords[i]} aborted.")
            exit()

        if plot:
            plt.figure(figsize=[12,10])
            plot_tuning(responses, freqs, start_pos, coords[i], starts[i], ends[i])
        if save:
            save_tuning(responses, freqs, start_pos, coords[i], starts[i], ends[i])
        if i == 0:
            mode_maps = np.zeros((len(coords),*responses.shape))
        mode_maps[i] = responses

    auto.webhook.send(f"Scan of {coords} COMPLETE")

    return mode_maps

def scan_multialignment(auto, coords, starts, ends, incrs, plot=True, save_plots=True, save_data=True,):
    '''
    Take several scans along coords[0], perturbing coords[1] after each scan
    
    ONLY WORKS FOR TWO PARAMS AT A TIME
    '''

    N_cycles = np.arange(starts[1],ends[1]+incrs[1],incrs[1]).size

    # set start of coords[1]
    kwarg = {coords[1]: starts[1]}
    auto.hexa.incremental_move(**kwarg)

    for frame in range(N_cycles):

        err,start_pos = auto.hexa.get_position()

        print(f"hexapod started cycle {frame}/{(ends[1]-starts[1])/incrs[1]} at {start_pos}")

        if err != 0:
            print(f'ERROR {err} with hexapod, exiting')
            auto.hexa.close()
            exit(err)
        
        seq = generate_single_axis_seq(coord=coords[0], incr=incrs[0], start=starts[0], end=ends[0])
        responses, freqs, collision = auto.tuning_scan_safety(seq, delay=0.2)

        if collision:
            auto.webhook.send("COLLISION! Scan aborted.")
            exit()

        if(responses is not None):
            if plot:
                plt.figure(figsize=[12,10])
                plot_tuning(responses, freqs, start_pos, coords[0], starts[0], ends[0])
                if save_plots:
                    plt.savefig(f"plots/dV_{start_pos[4]}X.png")
            if save_data:
                save_tuning(responses, freqs, start_pos, coords[0], starts[0], ends[0])

        kwarg = {coords[1]: incrs[1]}
        auto.hexa.incremental_move(**kwarg)

    kwarg = {coords[1]: -N_cycles*incrs[1]-starts[1]}
    auto.hexa.incremental_move(**kwarg)

    auto.webhook.send(f"Multiscan of {coords} COMPLETE")
    
def autoalign(auto, coords, margins, coarse_ranges, fine_ranges, N=20, max_iters=10, search_orders=None, plot_coarse=False, plot_fine=False, save=True, skip_coarse=False, start_ind=0, stop_ind=-1, harmon=9):
    '''
    Align automatically.

    takes a list of parameters, and the error margin to align to, and a max_iters
    '''

    # While not above max_iters,
        # For each param:
            # make a scan of N points on param axis
            # fit maximum frequency point of fundamental (quadratic)
            # move to that minimum
        # if all params in margin, break

    coord_lookup = np.array(['dX', 'dY', 'dZ', 'dU', 'dV', 'dW'])

    if search_orders is None:
        search_orders = ['fwd']*len(coords)

    starts = -coarse_ranges
    ends = -starts
    incrs = (ends-starts)/N

    search_range_coarse = 200

    aligned = np.array([False]*len(coords))

    _, start_pos = auto.hexa.get_position()

    freqs = auto.na.get_pna_freq()


    if not skip_coarse:
        deltas = np.zeros(len(coords))
        # first align each coord coarsely, all at once (since no iteration)
        for i, coord in enumerate(coords):
            if starts[i] == 0:
                # skip the coarse step for this coord
                continue 
            raw_responses = scan_one(auto, coord, starts[i], ends[i], incrs[i], plot=False, save=save)[:,start_ind:stop_ind]
            specs = analyse.fft_cable_ref_filter(raw_responses, harmon)
            fund_inds, skipped = analyse.get_fundamental_inds(specs,freqs,search_range=search_range_coarse, search_order=search_orders[i])
            param_vals = np.linspace(starts[i]+incrs[i]/2,ends[i]-incrs[i]/2,N+1) + start_pos[np.where(coord_lookup == coord)[0]]
            coarse_align_pos = param_vals[np.argmax(fund_inds)]
            deltas[i] = coarse_align_pos - start_pos[np.where(coord_lookup == coord)]
            
            if plot_coarse:
                plt.figure(figsize=[12,10])
                plot_tuning(specs, freqs, start_pos, coords[i], starts[i], ends[i])
                plt.plot(freqs[fund_inds]*1e-9, np.delete(param_vals,skipped), 'r.')

        if plot_coarse:
            plt.show()

        # do the coarse alignment in one shot
        command = {coord:deltas[i] for i, coord in enumerate(coords)}
        print(f'Aligning... {command}')
        auto.hexa.incremental_move(**command)

        print(f'coarse alignment COMPLETE (deltas: {deltas})')
    else:
        print('coarse alignmet SKIPPED')

    starts = -fine_ranges
    ends = -starts
    incrs = (ends-starts)/N
    # iterate to find fine alignment

    search_range_fine = 50

    phase_path = [[]*len(coords)]

    iter = 0
    while iter < max_iters and np.any(aligned == False):
        for i,coord in enumerate(coords):
            err, start_pos = auto.hexa.get_position()
            # can be expaned to different ranges for each coord.
            raw_responses = scan_one(auto, coord, starts[i], ends[i], incrs[i], plot=False, save=save)
            print(raw_responses.shape)
            print(start_ind, stop_ind)
            raw_responses = raw_responses[:,start_ind:stop_ind]
            specs = analyse.fft_cable_ref_filter(raw_responses, harmon)
            tp = analyse.get_turning_point(specs, coord, start_pos, starts[i], ends[i], incrs[i],search_range_fine, freqs, plot=plot_fine)     
            delta = tp - start_pos[np.where(coord_lookup == coord)[0]][0]
            print(f"{coord} tp at {tp}, delta of {delta}")
            if abs(delta) < margins[i]:
                aligned[i] = True
            else:
                aligned[i] = False
                command = {coord:delta}
                print(f'Adjusting {command}')
                auto.hexa.incremental_move(**command)

            if plot_fine:
                plt.show()
        iter += 1
    if iter >= max_iters:
        print('autoalignment FAILED, max iters reached')
        auto.webhook.send('Autoalign FAILED, exiting')
        exit(-1)
    else:
        print(f'autoalignment SUCCESS after {iter} iterations')
        auto.webhook.send(f'Autoalign SUCCESS after {iter} iterations')

def wide_z_scan(auto, zi, zf, N, align_count, plot=False, save=True):
    '''
    Since Z is tuned over a wide range, we want to autoalign on each step.
    Since the frequency of the fundamental will shift as we adjust Z, we need to be smart about
    where we look for it (that's why we need this special function)

    note intial (zi) and final (zf) positions are relative to positioner's location when starting
    '''

    fftfilt_harmonic = 60
    incr = (zf-zi)/N

    freqs = auto.na.get_pna_freq()
    freqs_ghz = freqs*1e-9

    responses = np.zeros((N, freqs.size))

    indices_per_ghz = freqs.size / ((freqs_ghz[-1]-freqs_ghz[0]))
    # all in frequency, GHz (doesn't change with resolution)
    autoalign_domain_r = 0.05

    # fit line to freq evolution of fundamental as a fct of Z
    # to predict where the fundamental will be next
    # these values must be measured. don't even think can be automated (easily)
    Zs = np.array([10.4, 18.5, 34.7, 42.8, 50.9, 63.05, 71.15])
    freq_points = [7.7394, 7.6991, 7.6189, 7.5818, 7.5410, 7.4833, 7.4441]
    p = np.polynomial.polynomial.Polynomial.fit(Zs,freq_points,1)
    # note: the fit is good but print(p)'s coeffs don't make sense

    # for autoalign
    align_coords = ['dX', 'dY', 'dV', 'dW']
    align_margins = [0.005,0.005, 0.005,0.005]
    align_coarse_ranges = np.array([0.1,0.3,0.1,0.1])
    align_fine_ranges = np.array([0.02,0.1,0.05,0.05])
    align_search_orders = ['fwd','rev','fwd','fwd']

    # decide when to autoalign based on align_count
    # want to space them out in the middle and do one at the beginning
    align_iters = np.linspace(0,N,align_count+1, dtype=int)[:-1]

    auto.pos.incremental_move(zi)

    # for i in range N
        # determine the range you should look for the fund in
        # autoalign limiting to that range (inefficient... could try only asking for the data you use)
        # take a spectrum

    for i in range(N):

        if i in align_iters:
            # get range to autoalign in
            Z = auto.pos.get_position()
            ind_center = int(indices_per_ghz * (p(Z) - freqs_ghz[0]))
            ind_r = int(indices_per_ghz * autoalign_domain_r)
            start_ind = ind_center - ind_r
            stop_ind = ind_center + ind_r

            skip_coarse = False
            if Z < 30:
                skip_coarse=True # not enough range of motion anyway

            # could try setting skip_coarse = True here to see if it still works
            autoalign(auto, align_coords, align_margins, align_coarse_ranges, align_fine_ranges, start_ind=start_ind, stop_ind=stop_ind, skip_coarse=skip_coarse, plot_coarse=True, plot_fine=True)

        freqs, response = read_spectrum(auto, harmon=60)

        if i == 0:
            responses = np.zeros((N, len(response)))
        responses[i] = response

    if plot:
        plt.figure(figsize=[8,6])
        plot_tuning(responses, freqs, [-1]*6, 'dZ', zi, zf)
    if save:
        save_tuning(responses, freqs, [-1]*6, 'dZ', zi, zf)

    # [-1]*6 for start pos since autoaligned each time so no good answer
    
    auto.webhook.send(f"Wide dZ scan COMPLETE")
    

    # save, plot, etc.


def read_spectrum(auto, harmon=None, save=True, plot=False, complex=False):

    freqs = auto.na.get_pna_freq()

    if complex:
        pna_func = auto.na.get_pna_complex_response
    else:
        pna_func = auto.na.get_pna_response

    if harmon is not None:
        response = np.vstack((pna_func(), np.zeros(len(freqs))))
        response = analyse.fft_cable_ref_filter(response, harmon=harmon, plot=plot)[0]
    else:
        response = pna_func()

    print(freqs)
    print(response)

    Zpos = auto.pos.get_position()

    if plot:
        plt.figure()
        if complex:
            plt.figure()
            plt.title("MAG")
            plt.plot(freqs, np.log10(np.abs(response)))
            plt.figure()
            plt.title("PHASE")
            plt.plot(freqs, np.angle(response))
        else:
            plt.plot(freqs,response, label=Zpos)
    if save:
        data_dir = "C:\\Users\\FTS\\source\\repos\\axion_detector\\tuning_data\\"
        now_str = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
        fname = f"{data_dir}{now_str}_zoomed_{Zpos}Z"
        np.save(fname, np.vstack((freqs,response)))

    return freqs, response
    

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

    pos = Positioner(host=args.pos_ip, username='Administrator', password=args.pos_password)
    hexa = HexaChamber(host=args.hex_ip, username='Administrator', password=args.hex_password,xps=pos.get_xps())
    na = na_tracer.NetworkAnalyzer()

    webhook = Webhook.from_url("https://discordapp.com/api/webhooks/903012918126346270/wKyx27DEes1nibOCvu1tM6T5F4zkv60TNq-J0UkFDY-9WyZ2izDCZ_-VbpHvceeWsFqF", adapter=RequestsWebhookAdapter())

    auto = AutoScanner(hexa, pos, na, webhook)
    
    '''
    coords = np.array(['dX', 'dY', 'dU', 'dV', 'dW'])
    starts = np.array([-0.1, -0.2, -0.6, -0.1, -0.1])
    ends = -1*starts
    incrs = 0.005*ends
    '''
    
    '''
    freqs = auto.na.get_pna_freq()
    response = np.vstack((auto.na.get_pna_response(), np.zeros(12801)))
    spec = analyse.fft_cable_ref_filter(response, harmon=9)[0]

    data_dir = "C:\\Users\\FTS\\source\\repos\\axion_detector\\tuning_data\\"
    now_str = datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    fname = f"{now_str}_zoomed_74.4Z"
    print(f"Saving to: {data_dir}\\{fname}.npy")
    np.save(f"{data_dir}{fname}", np.vstack((freqs,spec)))

    plt.plot(freqs*1e-9, spec)
    '''
    
    '''
    r = 2.2
    scan_one(auto, 'dY', -r, r, 0.1*r, plot=True,save=True)
    plt.show()
    exit()
    '''

    #autoalign(auto, ['dX', 'dY', 'dV', 'dW'], [0.005,0.005, 0.005,0.005], coarse_ranges=np.array([0.1,0.5,0.1,0.1]), fine_ranges=np.array([0.02,0.1,0.05,0.05]), search_orders=['fwd','rev','rev','fwd'], plot_coarse=True, plot_fine=False, skip_coarse=False)
    #webhook.send('Autoalign complete.')
    

    '''
    coords = np.array(['dX', 'dV'])
    starts = np.array([-0.1, -0.1])
    ends = -1*starts
    incrs = 0.1*ends
    '''
    #scan_many(auto, coords, starts, ends, incrs, plot=True, save=True)

    '''
    for i in range(5):
        autoalign(auto, ['dX', 'dV', 'dW'], [0.01,0.01,0.01], coarse_ranges=np.array([0.05,0.2,0.1]), fine_ranges=np.array([0.02,0.05,0.05]), search_orders=['fwd','fwd','rev'], plot_coarse=False, plot_fine=False, save=False)
        read_spectrum(auto, plot=False, save=True, harmon=9)
        auto.pos.incremental_move(1)

    auto.pos.incremental_move(-5)
    '''
    
    
    coord = 'dZ'
    start = -10
    end = 10
    incr = end/300
    scan_one(auto, coord, start, end, incr, plot=True, save=True)
    

    #read_spectrum(auto, harmon=None, save=True, plot=True, complex=True)

    #wide_z_scan(auto, 0, 91.4 - 10.4, 20, 3, plot=True)

    #plt.legend()
    plt.show()

    hexa.close()
    pos.close()

if __name__ == '__main__':
    main()