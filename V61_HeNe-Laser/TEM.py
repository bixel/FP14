#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

# TEM-00 Mode
x_00, I_00 = np.genfromtxt('data/TEM_00.txt', unpack=True)

def gauss(r, off, I_0, omega):
    return I_0 * np.exp(-2 * (r - off)**2/omega**2)

coeff, covar = curve_fit(gauss, x_00, I_00)
coeff_errs = np.sqrt(np.diag(covar))
offset = ufloat(coeff[0], coeff_errs[0])
I_0 = ufloat(coeff[1], coeff_errs[1])
omega = ufloat(coeff[2], coeff_errs[2])
print('offset: {:g}'.format(offset))
print('I_0: {:g}'.format(I_0))
print('omega: {:g}'.format(omega))

theory_xs = np.linspace(2, 7, 200)
theory_ys = gauss(theory_xs, *coeff)
plt.plot(theory_xs, theory_ys, 'r-')
plt.errorbar(x_00, I_00, yerr=0.05*I_00, fmt='b+')
plt.savefig('build/TEM_00.pdf')
plt.clf()


# TEM-01 Mode

def tem01(r, off, I_0, omega):
    return I_0 * ((r - off)/omega)**2 * gauss(r, off I_0, omega)

x_01, I_01 = np.genfromtxt('data/TEM_01.txt', unpack=True)

plt.errorbar(x_01, I_01, yerr=0.05*I_01, fmt='b+')
plt.savefig('build/TEM_01.pdf')
plt.clf()
