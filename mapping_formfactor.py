import numpy as np
import color_map # has data reading functions

fname = 'C:/Users/FTS/source/repos/axion_detector/field_mapping_data/20220831_132445.csv'

fres, deltas = color_map.read_deltas(fname, return_fres=True)

# df/f = -(eps-1)/2 * Vbead/Vcavity * E**2/mean(E**2)
# C = (sum(E dot z * d3x))**2 / (V * sum(E**2 * d3x))

# since we have 5x5x2 grid, make d3x 1/50 for V=1

d3x = 1/50

# hoping the prefactor on df = f * E**2 cancels somehow for now
# also assuming all E in the z direction, at least approx true for fund.

Es = np.sqrt(deltas / fres)
E2s = deltas / fres

C = np.sum(Es * d3x)**2 / np.sum(E2s * d3x)

print(C)
