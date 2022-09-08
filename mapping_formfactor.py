import numpy as np
import color_map # has data reading functions

fname = 'C:/Users/FTS/source/repos/axion_detector/field_mapping_data/20220831_133457.csv'

fres, deltas = color_map.read_deltas(fname, return_fres=True)

# df/f = -(eps-1)/2 * Vbead/Vcavity * E**2/mean(E**2)
# C = (sum(E dot z * d3x))**2 / (V * sum(E**2 * d3x))

# since we have 5x5x2 grid, make d3x 1/50 for V=1

d3x = 1/50

# the prefactor on df = E**2 is irrelevant for C it turns out (empirically, numerically)
# also assuming all E in the z direction, at least approx true for fund.
# I imagine there should be an abs around the deltas.

Es = np.sqrt(np.abs(deltas))

C = np.sum(Es * d3x)**2 / np.sum(Es**2 * d3x)

print(C)
