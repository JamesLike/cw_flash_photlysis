import attenuation_modeling_funcs2 as amf
import multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

print("Number of processors: ", mp.cpu_count())
width = 1.0
N0 = 1.20e15  # sample per cm-2
sig = 1.47e-16  # cross section per cm-2
phi = 0.41  # quantum yeidld of converstion
tstep = 1e-3  # In Seconds
tmax = 20e-3  # in seconds
ED = np.linspace(0.1, 40,  100)
resid=0.167


def multi_model(ED, phi, width, sig, tstep, tmax, N0, resid):
	pool = mp.Pool(mp.cpu_count())
	results_object = [pool.apply_async(amf.model, args=(EDi, phi, width, sig, tstep, tmax, N0, resid)) for EDi in ED]
	result_ED = np.asarray([r.get()[1] for r in results_object])
	result_percent = np.asarray([r.get()[0] for r in results_object])
	result_thermal = np.asarray([r.get()[2] for r in results_object])
	pool.close()
	pool.join()
	return result_ED, result_percent, result_thermal

def exponenial_func(x, a, b, c):
	return a * np.exp(-b * x) + c

model=multi_model(ED, phi, width, sig, tstep, tmax, N0, resid)
dat = np.loadtxt('/home/james/Documents/code/flash_dat2.dat')

plt.scatter(model[0], model[1], label='model')
plt.scatter(model[0], model[2]/np.max(model[2]),label='thermal')
plt.scatter(dat[:,0], dat[:,1],label='data')
plt.legend()
plt.show()