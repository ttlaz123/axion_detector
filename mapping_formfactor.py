import numpy as np
import color_map # has data reading functions

fname = 'C:/Users/FTS/source/repos/axion_detector/field_mapping_data/20220831_133457.csv'

fres, deltas = color_map.read_deltas(fname, return_fres=True)

perturb = 1e5

deltas2 = deltas - perturb

# df/f = -(eps-1)/2 * Vbead/Vcavity * E**2/mean(E**2)
# C = (sum(E dot z * d3x))**2 / (V * sum(E**2 * d3x))

# since we have 5x5x2 grid, make d3x 1/50
# each measured disk point gives the delta representative of 1/50th the volume of the cavity.

d3x = 1/50

# the prefactor on df = -(E**2) is irrelevant for C it turns out (empirically, numerically)
# also we assume all E is in the z direction, at least approx true for fundamental mode.

Es = np.sqrt(-1*deltas)
Es2 = np.sqrt(-1*deltas2)

# since d3x was fractional volume, no need to divide by d3x.
# you can make d3x whatever you want and divide C by d3x*Es.size if you want.
# The enitre physical volume is always taken into account.

C = np.sum(Es * d3x)**2 / np.sum(Es**2 * d3x)
C2 = np.sum((Es2 * d3x)**2 / np.sum(Es2**2 * d3x))

print(C,C2)
