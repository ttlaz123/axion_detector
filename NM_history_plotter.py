import numpy as np
import matplotlib.pyplot as plt

fname = "NM_histories\\20221010_115331_NM_history.npy"

history = np.load(fname) # (coord, nstep)

print(history)

plt.figure(figsize=(10,20))
plt.title("Best Position at each Iter")
coords = ["X", "Y", "U", "V", "W"]
for i in range(5):
    if i == 0:
        ax1 = plt.subplot(511+i)
    else:
        plt.subplot(511+i, sharex=ax1)
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

plt.tight_layout()
plt.show()