import attenuation_modeling_funcs2 as amf
import multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

print("Number of processors: ", mp.cpu_count())
width = 1.0
N0 = 1.20e15  # sample per cm-2
sig = 1.47e-16  # cross section per cm-2
phi = 0.3  # quantum yeidld of converstion
tstep = 1e-3  # In Seconds
tmax = 20e-3  # in seconds
residual = 0.1
dat = np.loadtxt('/home/james/Documents/code/flash_dat2.dat')


sig = np.zeros(len(dat))
for i in enumerate(dat):
	sig[i[0]]= i[1][1] / i[1][0]



# amf.model(1, 2, 1, 1.47e-16, 1e-3, 20e-3,1.20e15 )

def fitting_model(ED, phi, residual):
	val = np.empty_like(ED)
	for EDi in enumerate(ED):
		val[EDi[0]] = amf.model(EDi[1], phi, 1,    1.47e-16, 1e-3, 20e-3, 1.20e15, residual)
		#						E       phi  width sig       tstep tmax   N0
	return val


# amf.model(1, phi, 1, 1.47e-16, 1e-3, 20e-3,1.20e15 ) + offset
popt, pcov = curve_fit(fitting_model, dat[:, 0], dat[:, 1], p0=(phi, residual), sigma=sig , bounds=([0.1, 0.0], [0.99, 0.3]))
# popt, pcov = curve_fit(fitting_model, dat[:, 1], dat[:, 0], p0=(phi, offset))
fit = fitting_model(dat[:, 0], *popt)
plt.scatter(dat[:, 0], dat[:, 1], label='raw')
plt.scatter(dat[:, 0], fit, label='fit')
plt.show()
#

