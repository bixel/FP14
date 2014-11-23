#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from codecs import open

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
(open('build/tem00_x.tex', 'w', 'utf-8').
        write(r'$x_0 = \SI{{{:L}}}{{\milli\meter}}$'.
        format(offset)))
(open('build/tem00_i.tex', 'w', 'utf-8').
        write(r'$I_0 = \SI{{{:L}}}{{\milli\ampere}}$'.
        format(I_0)))
(open('build/tem00_omega.tex', 'w', 'utf-8').
        write(r'\omega = \SI{{{:L}}}{{\milli\meter}}'.
        format(omega)))


theory_xs = np.linspace(2, 7, 200)
theory_ys = gauss(theory_xs, *coeff)
plt.plot(theory_xs, theory_ys, 'r-', label='Fit')
plt.errorbar(x_00, I_00, yerr=0.05*I_00, fmt='b+', label='Messwerte')
plt.xlabel('$x/\mathrm{mm}$')
plt.ylabel('$I/\mathrm{mA}$')
plt.legend(loc='best')
plt.savefig('build/TEM_00.pdf')
plt.clf()


# TEM-01 Mode

def linear(x, a, b):
    return a * x + b

def tem01(r, off, I_0, omega):
    x = r - off
    return I_0 * (x/omega)**2 * np.exp(-2 * x**2/omega**2)

def tem01_improved(r, off, I_0, omega, a, b):
    x = r - off
    return linear(x, a, b) * tem01(r, off, I_0, omega)


x_01, I_01 = np.genfromtxt('data/TEM_01.txt', unpack=True)

coeff_01_imp, covar_01_imp = curve_fit(tem01_improved, x_01, I_01, p0=[17, 0.3, 1, -1, 1])
errs_01_imp = np.diag(np.sqrt(covar_01_imp))
offset = ufloat(coeff_01_imp[0], errs_01_imp[0])
I_0 = ufloat(coeff_01_imp[1], errs_01_imp[1])
omega  = ufloat(coeff_01_imp[2], errs_01_imp[2])
a = ufloat(coeff_01_imp[3], errs_01_imp[3])
b = ufloat(coeff_01_imp[4], errs_01_imp[4])

theory_xs = np.linspace(0, 35, 200)
theory_ys = tem01_improved(theory_xs, *coeff_01_imp)
plt.plot(theory_xs, theory_ys, 'r-', label='Fit')
plt.errorbar(x_01, I_01, yerr=0.05*I_01, fmt='b+',
        label='Messwerte')
plt.legend(loc='best')
plt.xlabel('$x/\mathrm{mm}$')
plt.ylabel('$I/\mathrm{mA}$')
plt.savefig('build/TEM_01_improved.pdf')
plt.clf()

print("""\
offset = {}
I_0    = {}
omega  = {}
a      = {}
b      = {}
""".format(offset, I_0, omega, a, b))

coeff_01, covar_01 = curve_fit(tem01, x_01, I_01, p0=[17, 1, 1])
errs_01 = np.diag(np.sqrt(covar_01))
offset = ufloat(coeff_01[0], errs_01[0])
I_0 = ufloat(coeff_01[1], errs_01[1])
omega  = ufloat(coeff_01[2], errs_01[2])
print("""\
offset = {}
I_0    = {}
omega  = {}
""".format(offset, I_0, omega, a, b))
(open('build/tem01_x.tex', 'w', 'utf-8').
        write(r'$x_0 = \SI{{{:L}}}{{\milli\meter}}$'.
        format(offset)))
(open('build/tem01_i.tex', 'w', 'utf-8').
        write(r'$I_0 = \SI{{{:L}}}{{\milli\ampere}}$'.
        format(I_0)))
(open('build/tem01_omega.tex', 'w', 'utf-8').
        write(r'\omega = \SI{{{:L}}}{{\milli\meter}}'.
        format(omega)))

theory_xs = np.linspace(0, 35, 200)
theory_ys = tem01(theory_xs, *coeff_01)
plt.plot(theory_xs, theory_ys, 'r-', label='Fit')
plt.errorbar(x_01, I_01, yerr=0.05*I_01, fmt='b+',
        label='Messwerte')
plt.legend(loc='best')
plt.xlabel('$x/\mathrm{mm}$')
plt.ylabel('$I/\mathrm{mA}$')
plt.savefig('build/TEM_01.pdf')
plt.clf()
