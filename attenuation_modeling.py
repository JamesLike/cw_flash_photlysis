import numpy as np
from matplotlib import pyplot as plt

dat=np.loadtxt('/home/james/Documents/code/flash_dat.dat')
plt.scatter(dat[:,1], dat[:,0])
plt.show()
# maxED = 20 # mJ/mm2
# stepED= 1
# width = 1.0
# N0    = 1.20e15
#
# sig   = 1.47e-16
# phi   = 0.3
# tstep = 1e-6  # In Seconds
# tmax  = 20e-3  # in seconds
#z = np.linspace(-0.1, 1.1, 1000)
	Ni = np.empty_like(z)
	I = np.empty_like(z)
	N = np.empty_like(z)
	I0 = ED * 100 / 4.42e-16 * tstep
	t = np.empty(int(tmax / tstep))

	for i in enumerate(t):
		t[i[0]] = tstep

	for i in enumerate(z):
		Ni[i[0]] = pop(N0*(1-residual), i[1], width)

	for k in enumerate(t):
		Intergral = 0
		for i in enumerate(z):
			if i[0] == 0:
				# print('This is ',I0)
				I[i[0]] = I0
				N[i[0]] = 0
			# Intergral=0
			else:
				N[i[0]] = Ni[i[0]] - phi * delI(Ni[i[0]], i[1], I[i[0] - 1], sig, width)
				I[i[0]] = I[i[0] - 1] - (z[i[0]] - z[i[0] - 1]) * delI(Ni[i[0]], i[1], I[i[0] - 1], sig, width)
				Intergral = Intergral + trap(z[i[0] - 1], z[i[0]], N[i[0] - 1], N[i[0]])
		Ni = N
# ED = np.linspace(0,maxED,int(maxED/stepED))
# percent = np.empty_like(ED)
#
# for k in enumerate(ED):
#
#     dI0dt = k[1] * 100 / 4.42e-16  # *10e-3 #Intensity per cm2 per s
#
#     def pop(N, z):
#         if z > width:
#             u = 0
#         elif z < 0:
#             u = 0
#         else:
#             u = 1
#         return N * u
#
#     def atten(N, z, I0, sig):
#         if z < 0:
#             v = I0
#         elif z > width:
#             v = I0 * np.exp(-sig * N0 * width)
#         else:
#             v = I0 * np.exp(-sig * N * z)  # Remember to change line above!!
#         return v
#
#     def delN(N, phi, sig, I):
#         return N * phi * sig * I
#
#     z = np.linspace(-0.1, 1.1, 100)
#     t = np.empty(int(tmax / tstep))
#     N = np.empty_like(z)
#     I = np.empty_like(z)
#     dN = np.empty_like(z)
#     Nf = np.empty_like(z)
#     Intergral = 0
#
#
#     # Set up time:
#     for i in enumerate(t):
#         t[i[0]] = tstep
#
#     for j in enumerate(t):
#         I0 = dI0dt*j[1]
#         if j[0] == 0:
#             for i in enumerate(z):
#                 N[i[0]] = pop(N0,i[1])
#                 I[i[0]] = atten(pop(N0,i[1]),i[1],I0,sig)
#                 dN[i[0]]= delN(N[i[0]],phi,sig,I[i[0]])
#                 Nf[i[0]]= N[i[0]] - dN[i[0]]
#
#         else:
#             N = Nf
#             for i in enumerate(z):
#                 I[i[0]] = atten(N[i[0]],i[1],I0,sig)
#                 I[i[0]] = atten(pop(N[i[0]],i[1]),i[1],I0,sig)
#                 dN[i[0]]= delN(N[i[0]],phi,sig,I[i[0]])
#                 Nf[i[0]]= N[i[0]] - dN[i[0]]
#     Integral=0
#     for i in enumerate(z):
#         if i[0] > 0:
#             Intergral = Intergral + (z[i[0]]-z[i[0]-1])*((Nf[i[0]]+Nf[i[0]-1])/2) # The trapeze rule - there must be a nicer way to do this..
#     percent[k[0]] = (Intergral / N0)
#     print("Per: ", (Intergral / N0))
#
# plt.plot(z, N / N0, label='N')
# plt.plot(z, I / I0, label='I')
# plt.plot(z, dN / N0, label='dN')
# plt.plot(z, Nf / N0, label='Nf')
# plt.legend()
# plt.show()
#
# plt.plot(ED,percent)
# plt.show()
#
# print("Per: ", (Intergral / N0))
# print("Raw: ", Intergral)
# print("N0:  ", N0)
#
#
#
# def exponenial_func(x, a, b, c):
# 	return a * np.exp(-b * x) + c
#
# popt, pcov = curve_fit(exponenial_func, ED, percent, p0=(1, 0.1, 0.1))
#
# fit = exponenial_func(ED, *popt)
#
# # fit = np.polyfit(ED, percent, 1)
# # pol = np.poly1d(fit)
# plt.plot(ED, percent, label='Calculation')
# plt.plot(ED, fit, label='fit')
# plt.show()
#
# print("Per: ", (Intergral / N0))
# print("Raw: ", Intergral)
# print("N0:  ", N0)
#
# # Per:  0.9939393939393938
# # Per:  0.8179495841042637
# # Per:  0.6701136792063358
# # Per:  0.5466604794992123
# # Per:  0.444139860350327
# # Per:  0.3594448499604154
# # Per:  0.2898168015969484
# # Per:  0.23283708527196156
# # Per:  0.1864088857616799
# # Per:  0.1487324520605611
# # Per:  0.11827665945975133
# # Per:  0.0937491587235212
# # Per:  0.07406679871421194
# # Per:  0.05832748156977455
# # Per:  0.04578417305772188
# # Per:  0.03582145201051079
# # Per:  0.02793473490006888
# # Per:  0.021712140891048084
# # Per:  0.01681885328930161
# # Per:  0.01298377005399889
# # Per:  0.01298377005399889
# # Raw:  15580524064798.668
# # # N0:   1200000000000000.0
# # Per:  0.9939393939393938
# # Per:  0.8187437376691322
# # Per:  0.6726643097481241
# # Per:  0.5514021522833694
# # Per:  0.45112955491013224
# # Per:  0.368488984631745
# # Per:  0.30057318229198154
# # Per:  0.24489251783916954
# # Per:  0.19933496829853486
# # Per:  0.16212300390215245
# # Per:  0.13177052138570863
# # Per:  0.10704192448894526
# # Per:  0.08691460342052758
# # Per:  0.07054542635585649
# # Per:  0.05724140876757908
# # Per:  0.04643443599702731
# # Per:  0.037659743109065245
# # Per:  0.03053776949642923
# # Per:  0.024758976065889445
# # Per:  0.020071219070492308
# # Per:  0.020071219070492308
# # Raw:  24085462884590.77
# # N0:   1200000000000000.0