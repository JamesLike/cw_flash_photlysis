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
resid = 0.1
ED = np.linspace(1, 40,  20)

pool = mp.Pool(mp.cpu_count())
results_object = [pool.apply_async(amf.model, args=(EDi, phi, width, sig, tstep, tmax, N0,resid)) for EDi in ED]
result_ED = np.asarray([r.get()[1] for r in results_object])
result_percent = np.asarray([r.get()[0] for r in results_object])
pool.close()
pool.join()


def exponenial_func(x, a, b, c):
	return a * np.exp(-b * x) + c

popt, pcov = curve_fit(exponenial_func, result_ED, result_percent, p0=(1, 0.1, 0.1))
fit = exponenial_func(result_ED, *popt)
plt.plot(result_ED, result_percent, label='Model')
plt.plot(result_ED,fit, label='fit')
plt.legend()
plt.show()
# amf.model(ED, phi, width, sig, tstep, tmax, N0)
