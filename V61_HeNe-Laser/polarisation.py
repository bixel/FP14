#! /usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from codecs import open
from textable import table


def sin(x, A, phi):
    return A * np.sin(x - phi)**2


theta, I = np.genfromtxt('data/polarisation.txt', unpack=True)

# close the circle
theta = np.append(theta, 360)
I = np.append(I, I[0])

coeffs, covar = curve_fit(sin, theta * (np.pi / 180), I,
                          p0=[0.1, 150*np.pi/180])
errs = np.sqrt(np.diag(covar))
I0 = ufloat(coeffs[0], errs[0])
phi = ufloat(coeffs[1], errs[1])
xs = np.linspace(0, np.max(theta), 200)
ys = sin(xs*(np.pi/180), *coeffs)

plt.polar(xs * np.pi / 180, ys, 'r-')
plt.polar(theta * np.pi / 180, I, 'bo--')
plt.savefig('build/polarisation_polar.pdf')
plt.clf()

(open('build/polarisation_I0.tex', 'w', 'utf-8').
 write(r'$I_0 = \SI{{{:L}}}{{\milli\ampere}}$'.
 format(I0)))
(open('build/polarisation_phi.tex', 'w', 'utf-8').
 write(r'$\varphi = \SI{{{:L}}}{{\degree}}$'.
 format(phi * 180 / np.pi)))

plt.plot(xs, ys, 'r-', label='Fit')

theta = theta[:-1]
I = I[:-1]
plt.errorbar(theta, I, yerr=0.05*I, fmt='bo', label='Messpunkte')
plt.xlabel(r'$\theta/^\circ$')
plt.ylabel(r'$I/\mathrm{mA}$')
plt.xlim(-10, 360)
plt.legend(loc='best')
plt.savefig('build/polarisation_lin.pdf')
plt.clf()

f = open('build/polarisation_tab.tex', 'w', 'utf-8')
f.write(table([r'$\theta/\si{\degree}$', r'$I/\si{\milli\ampere}$',
               r'$\theta/\si{\degree}$', r'$I/\si{\milli\ampere}$',
               r'$\theta/\si{\degree}$', r'$I/\si{\milli\ampere}$',
               r'$\theta/\si{\degree}$', r'$I/\si{\milli\ampere}$'],
              [theta[:9], I[:9],
               theta[9:18], I[9:18],
               theta[18:27], I[18:27],
               theta[27:36], I[27:36]]))
f.close()
