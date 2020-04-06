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
ED = np.linspace(0.1, 40,  20)
resid=0.1

#
# for EDi in ED:
# 	amf.model(EDi, phi, width, sig, tstep, tmax, N0)

def multi_model(ED, phi, width, sig, tstep, tmax, N0, resid):
	pool = mp.Pool(mp.cpu_count())
	results_object = [pool.apply_async(amf.model, args=(EDi, phi, width, sig, tstep, tmax, N0, resid)) for EDi in ED]
	result_ED = np.asarray([r.get()[1] for r in results_object])
	result_percent = np.asarray([r.get()[0] for r in results_object])
	pool.close()
	pool.join()
	return result_ED, result_percent

def exponenial_func(x, a, b, c):
	return a * np.exp(-b * x) + c


#result_ED, result_percent = multi_model(ED, phi, width, sig, tstep, tmax, N0)

#dat = np.loadtxt('/home/james/Documents/code/flash_dat.dat')
#popt, pcov =curve_fit(amf.model, dat[:,1], dat[:,0], p0=(phi, width, sig, tstep, tmax, N0))



#popt, pcov = curve_fit(multi_model, result_ED, result_percent, p0=(1, 0.1, 0.1))
# # fit = exponenial_func(result_ED, *popt)
# # plt.plot(result_ED,fit, label='fit')
# plt.legend()
# dat=np.loadtxt('/home/james/Documents/code/flash_dat.dat')
# plt.scatter(dat[:,1], dat[:,0])
# plt.show()

