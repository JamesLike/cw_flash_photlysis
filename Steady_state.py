import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# matplotpl.use('Qt5Agg')
#
# dir = '/media/james/data2_ext/data4/Other_data/RSFPs/Gel_Filtered/2017_12_19/'
# dattime = np.loadtxt(dir + '12198171_time_points.txt')
# dat = np.loadtxt(dir + '12198171_data.txt')
# datwav = np.loadtxt(dir + '12198171_wavelengths.txt')
#
# skytime = np.loadtxt(dir + '1219sk71_time_points.txt')
# skydat = np.loadtxt(dir + '1219sk71_data.txt')
# skywav = np.loadtxt(dir + '1219sk71_wavelengths.txt')
#test = np.loadtxt('/media/james/data2_ext/data4/Other_data/RSFPs/Gel_Filtered/2017_11_29/1129sk71_data.txt')

dir = '/home/james/Desktop/tmpspectdat/'
dattime = np.loadtxt(dir + '12198171_time_points.txt')
dat = np.loadtxt(dir + '12198171_data.txt')
datwav = np.loadtxt(dir + '12198171_wavelengths.txt')

skytime = np.loadtxt(dir + '1219sk71_time_points.txt')
skydat = np.loadtxt(dir + '1219sk71_data.txt')
skywav = np.loadtxt(dir + '1219sk71_wavelengths.txt')
test = np.loadtxt('/home/james/Desktop/tmpspectdat/1129sk71_data.txt')

meansky = np.mean(skydat[290:320, :], axis=0)


def trimsky(i, j):
	return [skytime[i:j] - skytime[i], meansky[i:j]]


dat1 = trimsky(29, 560)
dat2 = trimsky(628, 1235)
dat3 = trimsky(1264, 1804)
dat4 = trimsky(1834, 2442)
plt.plot(*dat1)


# plt.plot(dat2[0], dat2[1])
# plt.plot(dat3[0], dat3[1])
# plt.plot(dat4[0], dat4[1])

#
def func(t, a, b, d):
	# a = np.mean(meansky[25])
	# d = 0.015 #np.mean(np.mean(skydat[350:400,:],axis=0))
	# c = 13.9
	return (a - d) * np.exp(-b * (t)) + d


#
def fitplot(dat, label):
	t = np.linspace(0, 1000)
	val = curve_fit(func, dat[0], dat[1])
	plt.plot(t, func(t, *val[0]), label=label + ' Fit')
	plt.plot(*dat)
	return val


f1 = fitplot(dat1, 'skdat1')
f2 = fitplot(dat2, 'skdat2')
f3 = fitplot(dat3, 'skdat3')
f4 = fitplot(dat4, 'skdat4')

skyF = (f1[0][1] + f3[0][1]) / 2
skyB = (f2[0][1] + f4[0][1]) / 2
plt.legend()
# plt.show()
# plt.plot(t,func(t, *f1[0]))
plt.show()

meankir = np.mean(dat[290:320, :], axis=0)


def trimkir(i, j):
	return [dattime[i:j] - dattime[i], meankir[i:j]]


Kdat1 = trimkir(144, 770)
Kdat2 = trimkir(819, 959)
Kdat3 = trimkir(977, 1775)
Kdat4 = trimkir(1809, 2004)

k1 = fitplot(Kdat1, 'K1')
k2 = fitplot(Kdat2, 'K2')
k3 = fitplot(Kdat3, 'K3')
k4 = fitplot(Kdat4, 'K4')
plt.legend()
plt.show()

kirF = (k2[0][1] + k4[0][1]) / 2
kirB = (k1[0][1] + k3[0][1]) / 2  # trans to cis
print('Trans-Cis Ratio Kir/Sky: ', kirB / skyB)
print('Cis-Trans Ratio: Kir/Sky', kirF / skyF)

skyspect = [skywav[70:370], np.mean(skydat[70:370, 0:25], axis=1)]
kirospect = [datwav[70:370], np.mean(dat[70:370, 780:810], axis=1)]
testspec = [datwav[70:370], np.mean(test[70:370, 0:30], axis=1)]
skyspect[1] = (skyspect[1] - np.mean(skyspect[1][-30:-1])) / (
		np.mean(skyspect[1][12:22]) - np.mean(skyspect[1][-30:-1]))
kirospect[1] = (kirospect[1] - np.mean(kirospect[1][-30:-1])) / (
		np.mean(kirospect[1][12:22]) - np.mean(kirospect[1][-30:-1]))
testspec[1] = (testspec[1] - np.mean(testspec[1][-30:-1])) / (
		np.mean(testspec[1][12:22]) - np.mean(testspec[1][-30:-1]))
# plt.plot(*skyspect, label='This K')

lighton = np.mean(dat[70:400, 110:125], axis=1)
lightoff = np.mean(dat[70:400, 131:141], axis=1)
excitation = lightoff - lighton


def gaus(x, a, cent, b):
	return a * np.exp(-((x - cent) ** 2) / (2 * b ** 2))


sourcefit = curve_fit(gaus, skywav[70:400], excitation, p0=[0.001, 500, 30])
sourcedat = [skywav[70:370], gaus(skywav[70:370], *sourcefit[0])]
x = np.linspace(200, 600, 1000)
plt.plot(*testspec, label='Skylan-NS On', c='m'),plt.plot(*kirospect, label='RS-Kiiro On') , plt.plot(x,100 * gaus(x, *sourcefit[0]),label='Excitation LED Spectrum')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance (a.u.)')
plt.legend()
#plt.savefig("/home/james/Documents/figures/skylan_kiiro_On.svg")#',format='svg')
plt.show()
#plt.show()


def integrate(f1, f2):
	sum = 0
	for val in enumerate(f1[1]):
		sum = sum + val[1] * f2[1][val[0]]
	return sum

def integrate2(f1):
	delx = (f1[0][-1]-f1[0][0]) / len(f1[0])
	for i in enumerate(f1[1]):
		if i[0]==0: total = i[1]
		if i[0]==(len(f1[1])-1): total = total + i[1]
		else: total = total + 2*i[1]
	return delx*total


print('Kiiro ON overlap: ', integrate(kirospect, sourcedat))
print('Skylan ON overlap: ', integrate(skyspect, sourcedat))


def normalise(wav, dat, i, j):
	spect = [wav[70:370], np.mean(dat[70:370, i:j], axis=1)]
	val = (spect[1] - np.mean(spect[1][-30:-1])) / (np.mean(spect[1][12:22]) - np.mean(spect[1][-30:-1]))
	return [spect[0], val]


skyonlight = normalise(skywav, test, 1300, 1310)
skyonnon = normalise(skywav, test, 1280, 1290)
excitation2 = skyonlight[1] - skyonnon[1]
sourcefit2 = curve_fit(gaus, skywav[70:370], excitation2, p0=[0.04, 405, 1])
sourcedat2 = [skywav[70:370], gaus(skywav[70:370], *sourcefit2[0])]

skyoff = normalise(skywav, test, 1980,1999)#  450, 500)
kiirooff = normalise(skywav, dat, 1980, 2000)
plt.plot(*skyoff, label='Skylan-NS Off', c='m'), plt.plot(*kiirooff, label='RS-Kiiro Off'), plt.plot(sourcedat2[0], 10*sourcedat2[1], label='laser')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Absorbance (a.u.)')
plt.legend()
#plt.savefig("/home/james/Documents/figures/skylan_kiiro_Off.svg")#',format='svg')
plt.show()

sourcedat[1] = sourcedat[1] / np.trapz(sourcedat[1], sourcedat[0])

plt.plot(skyspect[0], skyspect[1]*sourcedat[1])
plt.show()


print('Kiiro OFF overlap: ', integrate(kiirooff, sourcedat2))
print('Skylan OFF overlap: ', integrate(skyoff, sourcedat2))

print('Trans-Cis Ratio Kir/Sky: ', kirB / skyB)
print('Cis-Trans Ratio: Kir/Sky', kirF / skyF)

print('Trans-Cis Actual rate scale (Kir/sky)',
	  (kirB / skyB) * (integrate(kiirooff, sourcedat2) / integrate(skyoff, sourcedat2)))
print('Cis-Trans Actual rate scale (Kir/sky)',
	  (kirF / skyF) * (integrate(kirospect, sourcedat) / integrate(skyspect, sourcedat)))

print('Kiiro absoption at 406 nm: ', kiirooff[1][(406 - 190 - 70)])
print('Kiiro Max absorption at ', kirospect[0][np.argmax(kirospect[1])], ' is ',np.max(kirospect[1]) )
print('Skylan Max absorption at ', skyspect[0][np.argmax(skyspect[1])], ' is ',np.max(skyspect[1]) )
print('Skylan absoption at 488 nm: ', skyspect[1][(488 - 190 - 70)])

