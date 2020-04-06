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


# amf.model(1, 2, 1, 1.47e-16, 1e-3, 20e-3,1.20e15 )

def fitting_model(ED, phi, residual):
	val = np.empty_like(ED)
	for EDi in enumerate(ED):
		val[EDi[0]] = amf.model(EDi[1], phi, 1, 1.47e-16, 1e-3, 20e-3, 1.20e15, residual)
	return val


# amf.model(1, phi, 1, 1.47e-16, 1e-3, 20e-3,1.20e15 ) + offset
popt, pcov = curve_fit(fitting_model, dat[:, 0], dat[:, 1], p0=(phi, residual), bounds=([0.1, 0.0], [0.5, 0.3]))
# popt, pcov = curve_fit(fitting_model, dat[:, 1], dat[:, 0], p0=(phi, offset))
fit = fitting_model(dat[:, 0], *popt)
plt.plot(dat[:, 0], dat[:, 1], label='raw')
plt.plot(dat[:, 0], fit, label='fit')
plt.show()

# fitting_model(60, 0.1, offset)


# def multi_model(ED, phi, width, sig, tstep, tmax, N0):
# 	pool = mp.Pool(mp.cpu_count())
# 	results_object = [pool.apply_async(amf.model, args=(EDi, phi, width, sig, tstep, tmax, N0)) for EDi in ED]
# 	result_ED = np.asarray([r.get()[1] for r in results_object])
# 	result_percent = np.asarray([r.get()[0] for r in results_object])
# 	pool.close()
# 	pool.join()
# 	return result_ED, result_percent

# #
# def exponenial_func(x, a, b, c):
# 	return a * np.exp(-b * x) + c

# popt, pcov = curve_fit(multi_model, result_ED, result_percent, p0=(1, 0.1, 0.1))
# # fit = exponenial_func(result_ED, *popt)
# # plt.plot(result_ED,fit, label='fit')
# plt.legend()
# dat=np.loadtxt('/home/james/Documents/code/flash_dat.dat')
# plt.scatter(dat[:,1], dat[:,0])
# plt.show()
