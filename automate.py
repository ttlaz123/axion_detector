# import stuff

# generate instructions: axis, step size, number of steps (each way?)

# loop n times:
#   take data
#   make incremental move
# then return and loop the other way (-incremental move)

# make plot


import time 

import nidaqmx

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
    
    
    taskname = 'Dev' + str(task_number)
    voltage_channel = '/'.join([taskname, channel])
    touching = False
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(voltage_channel)
        if(timeout is None):
            timeout = -1
        time_start = time.time()
        time_elapsed = time.time() - time_start
        while(time_elapsed < timeout and timeout > 0 and HEXSTATUS = 'scanning'):
            voltage_btw_plate_cavity = task.read()
            if(voltage_btw_plate_cavity < danger_volts):
                touching = True
                print("Plate and cavity are touching!")
                break 
            time_elapsed = time.time()-time_start

    return touching
        