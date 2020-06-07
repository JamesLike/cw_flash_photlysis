import numpy as np

def pop(N, z,width):
	if z > width:
		u = 0
	elif z < 0:
		u = 0
	else:
		u = 1
	return N * u


def atten(N, z, I0, sig, width):
	if z < 0:
		v = I0
	elif z > width:
		v = I0 * np.exp(-sig * width)
	else:
		v = I0 * np.exp(-sig * N * z)  # Remember to change line above!!
	return v


def delN(N, phi, sig, I):
	return N * phi * sig * I


def I0_rate(ED):
	return ED * 100 / 4.42e-16  # *10e-3 #Intensity per cm2 per s


def model(ED, phi, width, sig, tstep, tmax, N0):\
	dI0dt = I0_rate(ED)
	z = np.linspace(-0.1, 1.1, 100)
	t = np.empty(int(tmax / tstep))
	N = np.empty_like(z)
	I = np.empty_like(z)
	dN = np.empty_like(z)
	Nf = np.empty_like(z)
	Intergral = 0

	# Set up time:
	for i in enumerate(t):
		t[i[0]] = tstep

	for j in enumerate(t):
		I0 = dI0dt * j[1]
		if j[0] == 0:
			for i in enumerate(z):
				N[i[0]] = pop(N0, i[1], width)
				I[i[0]] = atten(pop(N0, i[1], width), i[1], I0, sig, width)
				dN[i[0]] = delN(N[i[0]], phi, sig, I[i[0]])
				Nf[i[0]] = N[i[0]] - dN[i[0]]

		else:
			N = Nf
			for i in enumerate(z):
				I[i[0]] = atten(N[i[0]], i[1], I0, sig, width)
				I[i[0]] = atten(pop(N[i[0]], i[1], width), i[1], I0, sig, width)
				I[i[0]] = atten(pop(N[i[0]], i[1], width), i[1], I0, sig, width)
				dN[i[0]] = delN(N[i[0]], phi, sig, I[i[0]])
				Nf[i[0]] = N[i[0]] - dN[i[0]]
	for i in enumerate(z):
		if i[0] > 0:
			Intergral = Intergral + (z[i[0]] - z[i[0] - 1]) * (
					(Nf[i[0]] + Nf[i[0] - 1]) / 2)  # The trapeze rule - there must be a nicer way to do this..
	print("Percent: ", (Intergral / N0))
	return (Intergral / N0, ED)
