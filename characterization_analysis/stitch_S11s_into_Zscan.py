import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import analyse as ana

from uncertainties import ufloat

fnames = ['2022-10-12-17-50-02_zoomed_24Z.npy', '2022-10-12-14-56-10_zoomed_30Z.npy', '2022-10-11-10-33-07_zoomed_50Z.npy', '2022-10-06-14-25-41_zoomed_70Z.npy', '2022-10-10-15-11-46_zoomed_90Z.npy', '2022-10-12-14-47-41_zoomed_92Z.npy']
Z_poss = np.array([24, 30, 50, 70, 90, 92])
fit_wins = np.array([700]*3+[400]+[500]+[200])

results = np.zeros((len(fnames), 4)) # (file, param [fres, dfres, Q, dQ]

for i, fname in enumerate(fnames):

    freqs, spec = np.load(f"tuning_data/{fname}")

    peaks, properties = find_peaks(-spec, prominence=0.5)
    win = slice(peaks[0]-fit_wins[i],peaks[0]+fit_wins[i])
    wspec = spec[win]
    wfreqs = freqs[win]

    popt, pcov = ana.get_lorentz_fit(wfreqs, wspec, get_cov=True)

    plt.figure()
    plt.subplot(211)
    plt.plot(freqs, spec, 'k.')
    plt.plot(wfreqs, ana.skewed_lorentzian(wfreqs, *popt), 'r--')
    plt.subplot(212)
    plt.plot(wfreqs, wspec-ana.skewed_lorentzian(wfreqs, *popt), 'k.')
    plt.show()

    results[i] = [popt[4], np.sqrt(pcov[4][4]), popt[5], np.sqrt(pcov[5][5])]



plt.figure()

p, cov = np.polyfit(Z_poss, results[:,0], deg=1, w=1/results[:,1], cov=True)

plt.subplot(211)
plt.errorbar(Z_poss, results[:,0], yerr=results[:,1], capsize=2, fmt='k.')
plt.plot(Z_poss, np.polyval(p, Z_poss), 'r--')
plt.subplot(212)
plt.errorbar(Z_poss, results[:,0]-np.polyval(p, Z_poss), yerr=results[:,1], capsize=2, fmt='k.')
plt.plot(Z_poss, 0*Z_poss, 'r--')

up = [ufloat(p[i], np.sqrt(np.diag(cov))[i]) for i in range(len(p))]
for uparam in up:
    print(uparam)

plt.figure()
plt.errorbar(Z_poss, results[:,2], yerr=results[:,3], capsize=2, fmt='k.')

uQ = [ufloat(results[i,2], results[i,3]) for i in range(len(results))]
print(np.mean(uQ))

plt.show()
