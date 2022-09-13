import numpy as np
import matplotlib.pyplot as plt
import color_map # has data reading functions

fname = 'C:/Users/FTS/source/repos/axion_detector/field_mapping_data/20220831_132445.csv'

fres, deltas = color_map.read_deltas(fname, return_fres=True)

#color_map.plot_deltas(np.sqrt(np.abs(deltas)))

perturbs = np.linspace(-1e6,1e6)

Cs = 0*perturbs
for i,perturb in enumerate(perturbs):
    print(perturb)
    deltasp = deltas + perturb

    # df/f = -(eps-1)/2 * Vbead/Vcavity * E**2/mean(E**2)
    # C = (sum(E dot z * d3x))**2 / (V * sum(E**2 * d3x))

    # since we have 5x5x2 grid, make d3x 1/50
    # each measured disk point gives the delta representative of 1/50th the volume of the cavity.

    d3x = 1/50

    # the prefactor on df = -(E**2) is irrelevant for C it turns out (empirically, numerically)
    # also we assume all E is in the z direction, at least approx true for fundamental mode.

    Esp = np.sqrt(-1*deltasp)
    Es = np.sqrt(-1*deltas)

    # since d3x was fractional volume, no need to divide by d3x.
    # you can make d3x whatever you want and divide C by d3x*Es.size if you want.
    # The enitre physical volume is always taken into account.

    Cs[i] = np.sum(Esp * d3x)**2 / np.sum(Esp**2 * d3x)
   
print(Cs)
plt.plot(perturbs, Cs, 'k.')
plt.axvline(0, color='r', ls='--')
plt.title("Form Factor Under Change in Baseline Frequency")
plt.ylabel("C")
plt.xlabel("Baseline Freq Shift (Hz)")
plt.show()