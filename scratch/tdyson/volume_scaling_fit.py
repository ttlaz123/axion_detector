import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def volumeAnalytical(f, A, B):
    return A*np.power(f, -1) - B*np.power(f, -2)

fs = [7.576653e9, 7.52777e9, 7.39191e9] # Hz
Vs = [0.002661, 0.00268024, 0.00273476] # m^3

popt = curve_fit(volumeAnalytical, fs, Vs, p0=[1e8,1e16])
print(popt)

fs_smooth = np.linspace(min(fs), max(fs), 1000)
plt.plot(fs, Vs, 'k.')
plt.plot(fs_smooth, volumeAnalytical(fs_smooth, *popt), 'r--')
print(popt)
plt.show()
