import numpy as np
import matplotlib.pyplot as plt

def plot_NM_history(history):
    plt.figure(figsize=(10,20))
    plt.title("Best Position at each Iter")
    coords = ["X", "Y", "U", "V", "W"]
    for i in range(5):
        if i == 0:
            ax1 = plt.subplot(511+i)
        else:
            plt.subplot(511+i, sharex=ax1)
            plt.subplots_adjust(hspace=0)
        if i != 4:
            plt.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
        else:
            plt.xlabel("Nelder-Mead Iteration Number")
        
        if coords[i] == "X" or coords[i] == "Y": # linear units
            plt.plot(history[i] - history[i][-1], 'k')
            plt.ylabel(f"{coords[i]} (mm)")
        else: # angular
            plt.plot((history[i] - history[i][-1])*3600, 'k')
            plt.ylabel(f"{coords[i]} (arcsec)")


def get_improvement_steps(history):

    # find where a NM iter actually changed one of the positions (because it improved FoM)
    # returns starting position and each new position that was chosen as best
    deltas = np.diff(history, axis=1)
    real_steps = np.where(np.sum(deltas**2, axis=0) != 0)[0] + 1 # +1 to get the position that resulted from the change, i.e. the better position
    real_steps = np.hstack((0,real_steps)) # want to know the original position too
    progress_hist = history[:,real_steps]

    # presumably the align succeeded, so the last position is the aligned position
    # choose that to be zero.

    progress_hist = (progress_hist.T - progress_hist[:,-1]).T
    
    return progress_hist

    

datadir = "/home/tdyson/coding/axion_detector/NM_histories"
fname = "20221011_102950_NM_history.npy"

fpath = f"{datadir}/{fname}"

history = np.load(fpath) # (coord, nstep)

print(get_improvement_steps(history))

plot_NM_history(history)
plt.show()


