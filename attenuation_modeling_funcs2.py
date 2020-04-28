import numpy as np
from matplotlib import pyplot as plt


def pop(N, z, width):
	if z > width:
		u = 0
	elif z < 0:
		u = 0
	else:
		u = 1
	return N * u


def delI(N, z, I0, sig, width):
	if z < 0.0:
		dI = 0
	elif z > width:
		dI = 0
	else:
		dI = N * sig * I0
	return dI


def trap(z0, z1, N0, N1):
	return (z1 - z0) * (N0 + N1) / 2


def atten(N, z, I0, sig, width):
	if z < 0:
		v = I0
	elif z > width:
		v = I0 * np.exp(-sig * width)
	else:
		v = I0 * np.exp(-sig * N * z)  # Remember to change line above!!
	return v


def I0_rate(ED):
	return ED * 100 / 4.42e-16  # *10e-3 #Intensity per cm2 per s


# width = 1.0
# N0 = 1e15  # 1.20e15
# dI0dt = I0_rate(10)# 1.13e15  #*10e-3 #Intensity per cm2 per s
# sig = 1.47e-16
# phi = 0.3  # 0.3
# tstep = 10e-4  # In Seconds
# tmax = 20e-3  # in seconds

def model(ED, phi, width, sig, tstep, tmax, N0, residual):
	z = np.linspace(-0.1, 1.1, 10000)
	Ni = np.empty_like(z)
	I = np.empty_like(z)
	N = np.empty_like(z)
	I0 = ED * 100 / 4.42e-16 * tstep
	# 192.33 mJ/mm2 is the unattenuated ..
	t = np.empty(int(tmax / tstep))
	thermal = np.empty_like(z)

	for i in enumerate(t):
		t[i[0]] = tstep

	for i in enumerate(z):
		Ni[i[0]] = pop(N0 * (1 - residual), i[1], width)

	for k in enumerate(t):
		Intergral = 0
		therm_Intergral = 0
		for i in enumerate(z):
			if i[0] == 0:
				# print('This is ',I0)
				I[i[0]] = I0
				N[i[0]] = 0
				thermal[i[0]]=  0
			# Intergral=0
			else:
				N[i[0]] = Ni[i[0]] - phi * delI(Ni[i[0]], i[1], I[i[0] - 1], sig, width)
				I[i[0]] = I[i[0] - 1] - (z[i[0]] - z[i[0] - 1]) * delI(Ni[i[0]], i[1], I[i[0] - 1], sig, width)
				thermal[i[0]]= (1-phi) * delI(Ni[i[0]], i[1], I[i[0] - 1], sig, width)
				therm_Intergral = therm_Intergral + trap(z[i[0] - 1], z[i[0]], thermal[i[0] - 1], thermal[i[0]])

				Intergral = Intergral + trap(z[i[0] - 1], z[i[0]], N[i[0] - 1], N[i[0]])
		Ni = N
	print('Conversion:', 1 - Intergral / N0)
	return (1 - Intergral / N0) -residual, ED #, therm_Intergral

# plt.plot(z,Integral)
# plt.plot(z, Ni, label='Ni')
# plt.plot(z, Intergral, label='Intergra')
# plt.plot(z, I, label='I')
# plt.plot(z, N, label='N')
# plt.ylim(0, 1.1)
# # plt.ylim(0,1.1)
# #plt.legend()
# plt.show()
