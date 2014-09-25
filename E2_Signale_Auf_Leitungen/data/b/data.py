from uncertainties import *
from uncertainties.umath import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

xval, a_short, a_long = np.loadtxt("data.txt", comments="%" ,unpack=True)
a_diff = [0 for i in range(len(a_long))]
alpha = [0 for i in range(len(a_long))]
alpha_e = [0 for i in range(len(a_long))]

for i in range(len(a_long)):
	a_diff[i] = a_long[i] / a_short[i]

for i in range(len(a_long)):
	alpha[i] = -20*np.log(a_diff[i])

for i in range(len(a_long)):
	alpha_e[i] = str(-20*log(ufloat(a_long[i],0.5) / ufloat(a_short[i],0.5)))

print(alpha_e)
fig = plt.figure()
plt.plot(xval, a_short, ".r", label="Messpunkte kurz")
plt.plot(xval, a_long, ".b", label="Messpunkte lang")
plt.ylabel("U[mV]")
plt.xlabel("f[kHz]")
plt.legend()
plt.savefig("alpha_zeit.pdf")
plt.close(fig)
fig = plt.figure()
plt.plot(xval, alpha, ".k", label="Dämpfungskonstanten")
plt.ylabel("Dämpfung[dB]")
plt.xlabel("f[kHz]")
plt.legend()
plt.savefig("alpha.pdf")
plt.close(fig)