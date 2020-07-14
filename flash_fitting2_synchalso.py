from matplotlib import pyplot as plt
import numpy as np
from scipy.odr import *
from scipy.optimize import curve_fit

############################
# Crystal data
flashCOL = 'silver' #'lightblue'  #'#285088'
flashCOL1 = 'black'
rfreeCOL = 'red' #'#c4196b'
rworkCOL = 'blue' #'#3d9eaa'
fobsCOL = 'm'  #'#f7982c'
extrapCOL = 'gold'#'#b2fc8d
############################

syncdat = np.loadtxt('/home/james/data/flash_photolysis/normalised_neg_int2.dat')
odsync = [0, 2.8, 2.11, 1.675, 1.13, 0.545]
delodsync = 0.2
def funcengerr(E,ND,delND,delE):
	if ND > 0:
		val = np.sqrt(10**(-2*ND)*((E**2)*(delND**2)+delE**2))
	else:
		val = 0
	return val

errod = [funcengerr(86.96,i,0.1,0.5) for i in odsync]

# def Ntemp(E,a, T ,c):
# 	return a - b*np.exp(-c*E)#

# def Ntemp(E,a,T,c):
# 	Ea = 91e3
# 	A = 1.28e13
# 	t= 1e-3
# 	residual = 1
# 	#return a - (b*np.exp(-(c)*E))#*np.exp(-t*A*np.exp(-Ea/(8.31*T))))
# 	return a - residual*np.exp(-c*E)*np.exp(t*A*(np.exp(-Ea/(8.31*T))))

# def Ntemp(E,a,T,c):
# 	Ea = 91e3
# 	A = 1.28e13
# 	t = 10e-3
# 	resid = 0.9
# 	return a - resid*np.exp(-t*A*np.exp(-Ea/(8.31*T)))*np.exp(-c*E)

# def Ntemp(E,a,b,c):
# 	return a - b*np.exp(-c*E)

def kth(T):
	Ea = 91e3
	A = 1.28e13
	t = 5e-3
	return t*A*np.exp(-Ea/(8.31*T))

def Ntemp(E,a,b,c):
	Ea = 91e3
	A = 1.28e13
	t = 5e-3
	return 1- b*np.exp(-c*E)

rfree , rfreec = curve_fit(Ntemp, syncdat[:,0],syncdat[:,1]/100)
rwork , rworkcc = curve_fit(Ntemp, syncdat[:,0],syncdat[:,2]/100)
fobs , fobsc = curve_fit(Ntemp, syncdat[:,0],syncdat[:,3]/100)
extrap , extrapc = curve_fit(Ntemp, syncdat[0:5,0],syncdat[0:5,4]/100)


E = np.linspace(0,20,1000)
plt.fill_between(Ebig,modelledlower,modelledupper,color=flashCOL,  alpha=0.5, label='Flash Photolysis ',edgecolor='None' )

plt.errorbar(syncdat[:,0],syncdat[:,1]/100, xerr=errod, label='R$_{Free}$',c=rfreeCOL, linestyle='None', marker='o', markersize=6, markerfacecolor=rfreeCOL, capsize=2)
plt.plot(E,Ntemp(E,*rfree),c=rfreeCOL, linestyle='dashed')#,label='R$_{Free}$ Fit')

plt.errorbar(syncdat[:,0],syncdat[:,2]/100, xerr=errod, label='R$_{Work}$',c=rworkCOL, linestyle='None', marker='^', markersize=6, markerfacecolor=rworkCOL, capsize=2)
plt.plot(E,Ntemp(E,*rwork),c=rworkCOL, linestyle='dashed')#,label='R$_{Work}$ Fit')

plt.errorbar(syncdat[:,0],syncdat[:,3]/100, xerr=errod, label='F$_{calc}$-F$_{obs}$ ', c=fobsCOL, linestyle='None', marker='s', markersize=6, markerfacecolor=fobsCOL, capsize=2)
plt.plot(E,Ntemp(E,*fobs),c=fobsCOL, linestyle='dashed')#,label='F$_C$-F$_{obs}$ Fit')

plt.errorbar(syncdat[:,0],syncdat[:,4]/100, xerr=errod, label='Extrapolated', c=extrapCOL, linestyle='None', marker='D', markersize=6, markerfacecolor=extrapCOL, capsize=2)
plt.plot(E,Ntemp(E,*extrap),c=extrapCOL, linestyle='dashed')#,label='Extrap Fit')


plt.xlabel('Energy Density mJ/mm$^{2}$')
plt.ylabel('Population (fraction)')
plt.xlim(-0.5,18)
plt.legend(loc=4)
#plt.savefig("/home/james/Documents/figures/fitting2.svg")#',format='svg')
plt.show()

# def tempincrease(fit):
# 	offset = fit[0] - fit[1] - 0.1
# 	Ea = 91e3
# 	A = 1.28e13
# 	t = 5e-3
# 	return -Ea/(8.31*np.log(np.log(offset/t)/A))
#
# def tempincrease(b):
# 	Ea = 91e3
# 	A = 1.28e13
# 	return -Ea/(8.31*np.log(b[1]/A))

def kth(offset, t):
	return np.log(offset)/(-t)

def tmp(Ea,A,k):
	return -Ea/(8.31*np.log(k/A))

def temperror(x,dx,A,dA):
	return np.sqrt(((dA**2*x**2)/(A**2*(np.log(A)**2))+(dx**2/(np.log(A)**2))))

def tempincrease(fit):
	#offset = (fit[0] + fit[1] - 1 / -1.1)
	b= fit[1]
	offset = b / 0.814
	if offset < 1:
		Ea = 91e3
		A = 1.28e13
		t = 5e-3
		mean = -Ea/(8.31*np.log((-np.log(offset)/t)/A)) - 298
		meanup = -(Ea+5e3)/(8.31*np.log((-np.log(offset)/t)/(A*1e-1))) - 298
		meandown= -(Ea - 5e3) / (8.31 * np.log((-np.log(offset) / t) / (A * 10))) - 298

		#meanup = -Ea/(8.31*np.log((-np.log(offset)/t)/(10*A))) - 298
		#meanlow = -(Ea-5e3) / (8.31 * np.log((-np.log(offset) / t) / A)) -298
		#meanhigh = -(Ea + 5e3) / (8.31 * np.log((-np.log(offset) / t) / A)) -298
		meansig = temperror(Ea,5e3,A,1e13)# (5/91) * (mean+298)
	else: mean = 1
	#return [mean , meansig]
	return [mean, meandown, meanup]



print('Temp increase:', tempincrease(rfree))
print('Temp increase:', tempincrease(rwork))
print('Temp increase:', tempincrease(fobs))
print('Temp increase:', tempincrease(extrap))

# print(rfree)
# print(rwork)
# print(fobs)
# print(extrap)
#
# Temp increase: 142.73162880546357
# Temp increase: 133.44767236881916
# Temp increase: 137.47790453346005
# Temp increase: 123.51698872484326
# Temp increase: 132.45783389454414
# Temp increase: 85.37163916278178
# Temp increase: 104.61231926991957
# Temp increase: 108.72973404840468

def printtabs(fitted,str):
	print(str,'\t','$',round(phi(fitted),3),'$','\t','$',round(tempincrease(fitted)[0],2),'\t',round(tempincrease(fitted)[1],2),'-',round(tempincrease(fitted)[2],2),'$')

print('Method \t $\phi$ \t $\Delta T$ (K) \t $\delta T$')
printtabs(rfree, '$R_{Free}$')
printtabs(rwork,'$R_{Work}$')
printtabs(fobs, '$F_{Calc}-F_{Obs}$')
printtabs(extrap,'$Extrapolated$')
print('Flash Photolysis\t','$',round(phi(myoutput.beta),3),'$', '\t N/A ')


