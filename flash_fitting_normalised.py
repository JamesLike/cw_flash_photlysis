from matplotlib import pyplot as plt
import numpy as np
from scipy.odr import *
from scipy.optimize import curve_fit






flashCOL = 'silver' #'lightblue'  #'#285088'
flashCOL1 = 'black'
rfreeCOL = 'red' #'#c4196b'
rworkCOL = 'blue' #'#3d9eaa'
fobsCOL = 'm'  #'#f7982c'
extrapCOL = 'gold'#'#b2fc8d


syncdat = np.loadtxt('/home/james/Downloads/normalised_neg_int.dat')


# def N2(E, a,  b, c):
# 	return 1 - (1*np.exp(-c*E))

def N2(E,k,a):
	flashfit = [0.85787821, 0.81629303, 0.44485131]
	t = 10e-3
	return a - a*np.exp(-flashfit[2]*E-(k*t))

def Temp(k):
	Ea = 85*1000
	A  = 1.28e13
	return -Ea / (8.31*np.log(k/A))

def phi(k):
	eps = 115000/3 #in M^-1cm-1
	return k * 4.9e-19 *6.022e23/(100 * np.log(10) * eps)


rfree , rfreec = curve_fit(N2, syncdat[:,0],syncdat[:,1]/100)
rwork , rworkcc = curve_fit(N2, syncdat[:,0],syncdat[:,2]/100)
fobs , fobsc = curve_fit(N2, syncdat[:,0],syncdat[:,3]/100)
extrap , extrapc = curve_fit(N2, syncdat[0:5,0],syncdat[0:5,4]/100)


plt.plot(syncdat[:,0],syncdat[:,1]/100, label='R$_{Free}$',c=rfreeCOL, linestyle='None', marker='o', markersize=6, markerfacecolor=rfreeCOL)
plt.plot(E,N2(E,*rfree),c=rfreeCOL, linestyle='dashed')#,label='R$_{Free}$ Fit')

plt.plot(syncdat[:,0],syncdat[:,2]/100, label='R$_{Work}$',c=rworkCOL, linestyle='None', marker='^', markersize=6, markerfacecolor=rworkCOL)
plt.plot(E,N2(E,*rwork),c=rworkCOL, linestyle='dashed')#,label='R$_{Work}$ Fit')

plt.plot(syncdat[:,0],syncdat[:,3]/100, label='F$_{calc}$-F$_{obs}$ ', c=fobsCOL, linestyle='None', marker='s', markersize=6, markerfacecolor=fobsCOL)
plt.plot(E,N2(E,*fobs),c=fobsCOL, linestyle='dashed')#,label='F$_C$-F$_{obs}$ Fit')
#
plt.plot(syncdat[:,0],syncdat[:,4]/100, label='Extrapolated', c=extrapCOL, linestyle='None', marker='D', markersize=6, markerfacecolor=extrapCOL)
plt.plot(E,N2(E,*extrap),c=extrapCOL, linestyle='dashed')#,label='Extrap Fit')
plt.xlim(0, 20)
plt.show()

print(fobs)