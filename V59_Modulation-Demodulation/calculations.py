# coding: utf-8
import numpy as np
from uncertainties import ufloat
from uncertainties import unumpy as unp
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

# Teil d)
t1s = unp.uarray([960, 1940, 2940, 3970, 5030], 5)
t2s = unp.uarray([1090, 2150, 3175, 4175, 5190], 5)
delta_ts = (t2s - t1s)[:3]
delta_ts *= 1e-9
for i in range(len(delta_ts)):
    delta_ts[i] /= i + 1
delta_t = delta_ts.mean()

omega_t = ufloat(981, 2) * 1e3
t_t = 1 / omega_t  # periodendauer des Trägers in ns
tmin = t_t - delta_t
tmax = t_t + delta_t
fmin = 1 / tmax
fmax = 1 / tmin
delta_f = (fmax - fmin) / 2
m = delta_f / omega_t

with open('build/t_t.tex', 'w') as f:
    f.write(r'\SI{{{:L}}}{{\nano\second}}'.format(t_t * 1e9))

with open('build/delta_t.tex', 'w') as f:
    f.write(r'\SI{{{:L}}}{{\nano\second}}'.format(delta_t * 1e9))

with open('build/delta_f.tex', 'w') as f:
    f.write(r'\SI{{{:L}}}{{\kilo\hertz}}'.format(delta_f / 1e3))

with open('build/m.tex', 'w') as f:
    f.write(r'\num{{{:L}}}'.format(m))

# Teil e)
# Abhängigkeit phase-Spannung
T, U = np.genfromtxt('am-demodulation-e.txt', unpack=True)


# fit
def func(x, a, b, c, d):
    """ Simple cosine function
    """
    return a * np.cos(b * x + c) + d


popt, pocov = cf(func, T, U, p0=[150, 0.01, -1, 0])

plt.plot(T, U, 'r+', label='Messpunkte')
# plt.errorbar(T, U, yerr=0.1, label='Messpunkte', fmt='r+')
ts = np.linspace(np.amin(T), np.amax(T), 200)
plt.plot(ts, func(ts, *popt), label='Fit')
plt.xlabel(r'$T/\mathrm{ns}$')
plt.xlim(-5, 95)
plt.ylabel(r'$U/\mathrm{mV}$')
plt.legend(loc='best')
plt.savefig('build/demodulation-cosinus.pdf')
plt.clf()
