from matplotlib import pyplot as plt
import numpy as np
from scipy.odr import *
from scipy.optimize import curve_fit
#dat = np.loadtxt('/media/james/data2_ext/data4/flash_photolysis/2019_12_05/analysed.dat')

dat2=np.loadtxt('/home/james/Documents/data/mean.dat')
plt.xlabel('Energy Density mJ/mm$^{2}$')
plt.ylabel('Population (fraction)')


# def N(a, E):
# 	return a[0] - (a[1]*np.exp(-(a[2])*E))

def N(a, E):
	return a[0] - (a[1]*np.exp(-(a[2])*E))

#$$N = \frac{[Cis]}{[Trans]_0}= \frac{[Cis]_0}{[Tans]_0} - [Cis]_0 \exp{(-k_{Switch}E)}  $$
#-E \epsilon_{405 nm} \phi_{FWD} \times 100 \ln{10}

exp = Model(N)
mydata = RealData(dat2[:,0], dat2[:,1],sx=dat2[:,2], sy=dat2[:,3])
myodr = ODR(mydata, exp, beta0=[0.5,0.5,0.5])
myoutput = myodr.run()
myoutput.pprint()

Ebig = np.linspace(0,80,1000)
modelled = N(myoutput.beta,Ebig)
def argadjust(beta,error):
	a = np.empty(3)
	a[0] = beta[0] + 1.96 * error[0]
	a[1] = beta[1] - 1.96 * error[1]
	a[2] = beta[2] + 1.96 * error[2]
	return a #[a[0], a[1], a[2]]

modelledupper = N((argadjust(myoutput.beta, myoutput.sd_beta)),Ebig)
modelledlower = N((argadjust(myoutput.beta, -myoutput.sd_beta)),Ebig)
plt.plot(Ebig, modelled)
plt.fill_between(Ebig,modelledlower,modelledupper, alpha=0.5, label='95% Confidence', color='silver')
plt.errorbar(dat2[:,0], dat2[:,1], yerr=dat2[:,3], xerr=dat2[:,2],fmt='.', lines=None, c='g', label='Flash Photolysis', elinewidth=0.4, capsize=2)
plt.legend()
plt.show()
#plt.savefig("/home/james/Documents/figures/flash.svg")#',format='svg')

def phi(k):
	eps = 31185 #115000/3 #in M^-1cm-1
	return k * 4.9e-19 *6.022e23/(100 * np.log(10) * eps)

# print('Yield is: ', phi(myoutput.beta[2]))

syncdat = np.loadtxt('/home/james/Downloads/normalised_neg_int.dat')


def N2(E, a,b,c):
	return a - (b*np.exp(-c*E))

E = np.linspace(0,20,1000)


rfree , rfreec = curve_fit(N2, syncdat[:,0],syncdat[:,1]/100)
rwork , rworkcc = curve_fit(N2, syncdat[:,0],syncdat[:,2]/100)
fobs , fobsc = curve_fit(N2, syncdat[:,0],syncdat[:,3]/100)
extrap , extrapc = curve_fit(N2, syncdat[0:5,0],syncdat[0:5,4]/100)
modelledupper = N((argadjust(myoutput.beta, myoutput.sd_beta)),E)
modelledlower = N((argadjust(myoutput.beta, -myoutput.sd_beta)),E)


fobs, fobc = curve_fit(N2, syncdat[:,0],syncdat[:,3]/100)
flashCOL = 'silver' #'lightblue'  #'#285088'
flashCOL1 = 'black'
rfreeCOL = 'red' #'#c4196b'
rworkCOL = 'blue' #'#3d9eaa'
fobsCOL = 'm'  #'#f7982c'
extrapCOL = 'gold'#'#b2fc8d'



#plt.errorbar(dat2[:,0], dat2[:,1], yerr=dat2[:,3], xerr=dat2[:,2],fmt='.', lines=None, c=flashCOL1, label='Flash Photolysis', elinewidth=1, capsize=2, alpha=1)
plt.fill_between(E,modelledlower,modelledupper,color=flashCOL,  alpha=0.5, label='Flash Photolysis ',edgecolor='None' )
#plt.plot(E, N(myoutput.beta,E), c='black', alpha=0.2, linestyle='dashed')


plt.plot(syncdat[:,0],syncdat[:,1]/100, label='R$_{Free}$',c=rfreeCOL, linestyle='None', marker='o', markersize=6, markerfacecolor=rfreeCOL)
plt.plot(E,N2(E,*rfree),c=rfreeCOL, linestyle='dashed')#,label='R$_{Free}$ Fit')

plt.plot(syncdat[:,0],syncdat[:,2]/100, label='R$_{Work}$',c=rworkCOL, linestyle='None', marker='^', markersize=6, markerfacecolor=rworkCOL)
plt.plot(E,N2(E,*rwork),c=rworkCOL, linestyle='dashed')#,label='R$_{Work}$ Fit')

plt.plot(syncdat[:,0],syncdat[:,3]/100, label='F$_{calc}$-F$_{obs}$ ', c=fobsCOL, linestyle='None', marker='s', markersize=6, markerfacecolor=fobsCOL)
plt.plot(E,N2(E,*fobs),c=fobsCOL, linestyle='dashed')#,label='F$_C$-F$_{obs}$ Fit')
#
plt.plot(syncdat[:,0],syncdat[:,4]/100, label='Extrapolated', c=extrapCOL, linestyle='None', marker='D', markersize=6, markerfacecolor=extrapCOL)
plt.plot(E,N2(E,*extrap),c=extrapCOL, linestyle='dashed')#,label='Extrap Fit')

#plt.plot(dat2[:,0],dat2[:,1], alpha=0.2, linestyle='None', marker='.')
plt.xlim([-0.5, 15])

#plt.plot(syncdat[:,0], np.mean(syncdat[:,1:4],axis=1)/100)

plt.xlabel('Energy Density mJ/mm$^{2}$')
plt.ylabel('Population (fraction)')
plt.legend(loc=4)
plt.show()
#plt.savefig("/home/james/Documents/figures/fitting.svg")#',format='svg')

def T(k,A,Ea):
	return -Ea/(np.log(k/A)*8.31)
Ea = 91e3
A  = 1.28e13




#
# print('Rfree=',phi(rfree[2]))
# print('Rfree=',phi(rfree[2])-phi(myoutput.beta[2]))
# print (T(phi(rfree[2]) - phi(myoutput.beta[2]) , A, Ea))
#
# print('Rwork=',phi(rwork[2]))
# print('Rwork=',phi(rwork[2])-phi(myoutput.beta[2]))
# print (T(phi(rwork[2]) - phi(myoutput.beta[2]) , A, Ea))
#
# print('fobs=',phi(fobs[2]))
# print('fobs=',phi(fobs[2])-phi(myoutput.beta[2]))
# print (T(phi(fobs[2]) - phi(myoutput.beta[2]) , A, Ea))
#
# print('Extrap=',phi(extrap[2]))
# print('Extrap=',phi(extrap[2])-phi(myoutput.beta[2]))
# print (T(phi(extrap[2]) - phi(myoutput.beta[2]) , A, Ea))
#
# def N3(E,  b, c):
# 	return 1 - (b*np.exp(-c*E))
# dat2[:,1] = dat2[:,1]/np.max(dat2[:,1])
# modelled = N(myoutput.beta,E)
#
# plt.plot(E,np.gradient(N2(E,*rfree)),c=rfreeCOL, linestyle='dashed',label='R$_{Free}$ Fit')
# plt.plot(E,np.gradient(N2(E,*rwork)),c=rworkCOL, linestyle='dashed',label='R$_{Work}$ Fit')
# plt.plot(E,np.gradient(N2(E,*fobs)),c=fobsCOL, linestyle='dashed', label='F$_C$-F$_{obs}$ Fit')
# plt.plot(E,np.gradient(modelled), label='Model')
# plt.legend()
# plt.show()

# test , tests = curve_fit(N3, dat2[:,0], dat2[:,1])
# plt.plot(dat2[:,0], dat2[:,1])
# plt.plot(Ebig,N3(Ebig, *test))
# plt.show()
#
#
# exp = Model(N)
# mydata = RealData(dat2[:,0], dat2[:,1],sx=dat2[:,2], sy=dat2[:,3])
# myodr = ODR(mydata, exp, beta0=[0.5,0.5,0.5])
# myoutput = myodr.run()
# myoutput.pprint()
#
# Ebig = np.linspace(0,80,1000)
# modelled = N(myoutput.beta,Ebig)
#
# plt.plot(Ebig,modelled)
# plt.plot(dat2[:,0], dat2[:,1])
# plt.show()

flashfit = myoutput.beta
crystfit = [rfree, rwork, fobs, extrap]
#
def Ntemp(E,T):
	Ea = 91e3
	A = 1.28e13
	t= 1e-3
	a = myoutput.beta
	return a[0] - (a[1]*np.exp(-(a[2])*E)*np.exp(-t*A*np.exp(-Ea/(8.31*T))))
#
# def N4(E,T):
# 	return N(flashfit,E)*Ntemp(T)
#
# def N(a, E):
# # 	return a[0] - (a[1]*np.exp(-(a[2])*E))
#
# fit1 = curve_fit(Ntemp, syncdat[:,0],syncdat[:,3]/100, bounds=([100, 500]))
#
# plt.plot()
# plt.plot(E,Ntemp(E,*fit1[0]),label='Thermal')
# plt.plot(E,N2(E,*fobs),c=rfreeCOL, linestyle='dashed')#,label='R$_{Free}$ Fit')
# plt.plot(syncdat[:,0],syncdat[:,3]/100, label='F$_{calc}$-F$_{obs}$ ', c=fobsCOL, linestyle='None', marker='s', markersize=6, markerfacecolor=fobsCOL)
# plt.plot(E,N(myoutput.beta,E), label='normal')
# #plt.plot(E,N4(E,340), label='Thermal')
# plt.legend()
# plt.show()
# print(fit1)


