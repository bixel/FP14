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

plt.plot(xval, a_short, ".r")
plt.plot(xval, a_long, ".b")
plt.show()
plt.plot(xval, alpha, ".k")
plt.show()