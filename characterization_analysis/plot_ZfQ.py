import numpy as np
import matplotlib.pyplot as plt

fpath = "20221003_ZfQ_no_aligns.npy"

ZfQ = np.load(fpath)

plt.errorbar(ZfQ[:,0], ZfQ[:,1], yerr=ZfQ[:,2], capsize=2, fmt='k.')
plt.legend()
plt.figure()
plt.plot(ZfQ[:,0], ZfQ[:,3], yerr=ZfQ[:,4], capsize=2, fmt='k.')
plt.legend()
plt.show()