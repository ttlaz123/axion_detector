import numpy as np
import matplotlib.pyplot as plt

fname = "20221010_115331_NM_history.npy"

history = np.load(fname)

plt.figure()
plt.title("Best Solution at each Iter")
coords = ["X", "Y", "U", "V", "W"]
for i in range(5):
    if i == 0:
        ax1 = plt.subplot(511+i)
    else:
        plt.subplot(511+i, sharex=ax1)
    plt.plot(history[i])
    plt.ylabel(coords[i])

plt.show()