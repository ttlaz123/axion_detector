import numpy as np
import matplotlib.pyplot as plt

import polyplotter as pp

bkgnd_fname = "2023-02-10-14-10-45_zoomed_NoneZ.npy"
aligned_fname = "2023-02-10-14-06-10_zoomed_NoneZ.npy"
towards_fname = "2023-02-10-14-08-47_zoomed_NoneZ.npy"
away_fname = "2023-02-10-14-09-44_zoomed_NoneZ.npy"

fb, s11b = pp.load_spec(bkgnd_fname)
fa, s11a = pp.load_spec(away_fname)

s11r = s11a / s11b
'''
plt.figure()
plt.plot(np.real(s11r), np.imag(s11r), 'k.')

plt.figure()
plt.plot(fa, np.log10(np.abs(s11r)), 'k.')

plt.figure()
plt.plot(fa, np.unwrap(np.angle(s11r)), 'k.')
'''
plt.figure()
pp.plot_s11(fa, s11r, fit=True)

plt.show()
