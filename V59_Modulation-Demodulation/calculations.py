# coding: utf-8
import numpy as np
from uncertainties import ufloat
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as cf


# Teil c)
# Momentanspannung
offset = ufloat(16, 0.1)
U1 = ufloat(83.5, 0.1) + offset
U2 = ufloat(34, 0.1) + offset

m1 = (U1 - U2) / (U1 + U2)
print(m1)
with open('build/m1.tex', 'w') as f:
    f.write(r'\num{{{:L}}}'.format(m1))

# Spektrum
UT = ufloat(4.7, 0.1)
U1 = ufloat(1.57, 0.01)
U2 = ufloat(1.56, 0.01)

m2 = (U1 + U2) / UT
print(m2)
with open('build/m2.tex', 'w') as f:
    f.write(r'\num{{{:L}}}'.format(m2))

# Teil e)
# Abhängigkeit phase-Spannung
T, U = np.genfromtxt('am-demodulation-e.txt', unpack=True)


# fit
def func(x, a, b, c, d):
    """ Simple cosine function
    """
    return a * np.cos(b * x + c) + d


popt, pocov = cf(func, T, U, p0=[150, 0.01, -1, 0])

# plt.plot(T, U, 'r+', label='Messpunkte')
# plt.errorbar(T, U, yerr=0.1, label='Messpunkte', fmt='r+')
ts = np.linspace(np.amin(T), np.amax(T), 200)
plt.plot(ts, func(ts, *popt), label='Fit')
plt.xlabel(r'$T/\mathrm{ns}$')
plt.xlim(-5, 95)
plt.ylabel(r'$U/\mathrm{mV}$')
plt.legend(loc='best')
plt.savefig('build/demodulation-cosinus.pdf')
plt.clf()
