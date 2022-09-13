import numpy as np
import matplotlib.pyplot as plt
from comsol_formfactor_processing import read_comsol_integrations

data_path="form_factor_data"
fname = "aligned_all_components_other_norms_wf.txt"

fullname = f"{data_path}/{fname}"

cdat = read_comsol_integrations(fullname, colnames=['freq', 'ez', 'ey', 'ex', 'e2', 'v', 'norme_man', 'norme_auto'])

fs = cdat['freq']
ez = cdat['ez']
ex = cdat['ex']
ey = cdat['ey']
e2 = cdat['e2']
v = cdat['v']

plt.title("Fraction of E field in Z direction, norm calculated in COMSOL")
plt.plot(fs, (np.abs(ex)/cdat['norme_auto']), label="X")
plt.plot(fs, (np.abs(ey)/cdat['norme_auto']), label="Y")
plt.plot(fs, (np.abs(ez)/cdat['norme_auto']), label="Z")
plt.plot(fs, (np.sqrt((np.abs(ez)/cdat['norme_auto'])**2+(np.abs(ey)/cdat['norme_auto'])**2+(np.abs(ex)/cdat['norme_auto'])**2)), label="SUM")
plt.ylabel("Ez/norm(E)")
plt.xlabel("Frequency (GHz)")
plt.legend()
plt.figure()
plt.title("Fraction of E field in Z direction, norm calculated from components")
plt.plot(fs, (np.abs(ex)/np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)), label="X")
plt.plot(fs, (np.abs(ey)/np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)), label="Y")
plt.plot(fs, (np.abs(ez)/np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2)), label="Z")
plt.plot(fs, (np.sqrt((np.abs(ex)/np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2))**2+(np.abs(ey)/np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2))**2+(np.abs(ez)/np.sqrt(np.abs(ex)**2 + np.abs(ey)**2 + np.abs(ez)**2))**2)), label="SUM")
plt.ylabel("Ez/norm(E)")
plt.xlabel("Frequency (GHz)")
plt.legend()
plt.figure()
plt.title("Magnitudes of E field components")
plt.plot(fs, np.abs(ex), label="X")
plt.plot(fs, np.abs(ey), label="Y")
plt.plot(fs, np.abs(ez), label="Z")
plt.ylabel("integrated E (V * m^2)")
plt.xlabel("Frequency (GHz)")
plt.legend()
plt.show()


