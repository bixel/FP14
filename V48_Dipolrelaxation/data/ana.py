#! /usr/bin/env python3

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from scipy import constants
from scipy.optimize import curve_fit
from scipy.integrate import quad, trapz

from textable import table

from uncertainties import ufloat
from uncertainties.unumpy import exp

T1, I1 = np.genfromtxt('set1.txt', unpack=True)
T2, I2 = np.genfromtxt('set2.txt', unpack=True)
T1 = constants.C2K(T1)
T2 = constants.C2K(T2)
I1_cleaned, I2_cleaned = [], []

min1, max1 = 260, 285
min2, max2 = 250, 267

plotdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '../build/plots/')
texdir = os.path.join(plotdir, '../tex/')


# clear data
def linear_fit(T, A, B):
    return A*T + B


def exp_fit(T, A, B, T0, I0):
    return A * np.exp(B * (T - T0)) + I0


def mark_lower_cut(pltobj=plt):
    pltobj.vlines(min1, *pltobj.ylim(), colors='b', linestyle='dashed')
    pltobj.vlines(min2, *pltobj.ylim(), colors='g', linestyle='dashed')


def mark_fit_area(xlim, ylim, plt=plt, **kwargs):
    ax = plt.gca()
    ax.add_patch(Rectangle(
        (xlim[0], ylim[0]),
        xlim[1] - xlim[0],
        ylim[1] - ylim[0],
        **kwargs))

# t1, t2 = [], []
#
# for t, T, name in [[t1, T1, 'set1'], [t2 / 60, T2, 'set2']]:
#     var, covar = curve_fit(linear_fit, T, range(len(T)))
#     errs = np.sqrt(np.diag(covar))
#     b = ufloat(var[0], errs[0])
#     plt.plot(T, 'g.')
#     xs = np.linspace(np.min(t), np.max(t))
#     plt.plot(xs, linear_fit(xs, *var))
#     plt.savefig('{}b_fit_{}.pdf'.format(plotdir, name))
#     plt.clf()


for T, I, selection, p0, name, ff in [
        [
            T1,
            I1,
            ((T1 > 245) & (T1 < 275) | (T1 > 305)),
            [1, 1, constants.C2K(0), 5],
            'set1',
            exp_fit,
        ],
        [
            T2,
            I2,
            ((T2 > 280) & (T2 < 292) | (T2 > 245) & (T2 < 260)),
            None,
            'set2',
            linear_fit,
        ]]:
    fit_function = ff
    var, _ = curve_fit(fit_function, T[selection],
                       I[selection], p0=p0)
    I_cleaned = I - fit_function(T, *var)
    I_min = np.min(I_cleaned)
    I_cleaned -= I_min
    print(I_min)
    if name == 'set1':
        I1_cleaned = I_cleaned
    else:
        I2_cleaned = I_cleaned
    with open('{}data-{}.tex'.format(texdir, name), 'w') as f:
        f.write(table(
            3*[r'$T/\si{\kelvin}$', r'$I/\si{\pico\ampere}$',
               r'$I_\mathrm{cl}/\si{\pico\ampere}$'],
            [
                np.around(T[:23], 1), np.around(I[:23], 2),
                np.around(I_cleaned[:23], 2),
                np.around(T[23:46], 1), np.around(I[23:46], 2),
                np.around(I_cleaned[23:46], 2),
                np.around(T[46:], 1), np.around(I[46:], 2),
                np.around(I_cleaned[46:], 2),
            ]
        ))
    xs = np.linspace(240, 340)
    plt.plot(xs, fit_function(xs, *var), label='fit')
    plt.plot(T[~selection], I[~selection], 'b.',
             label='Data (ignored for fit)')
    plt.plot(T[selection], I[selection], 'g.', label='Data (used for fit)')
    plt.plot(T, I_cleaned - I_min, 'r.', label='Cleaned Data')
    plt.plot(xs, [-I_min] * len(xs), label='Offset')
    plt.xlim(240, 320)
    plt.ylim(0, 30)
    plt.xlabel(r'$T$ / K')
    plt.ylabel(r'$I$ / pA')
    # if p0:
    #     plt.plot(xs, fit_function(xs, *p0), label='fitfunction')
    plt.legend(loc='best')
    plt.savefig('{}cleaned_data-{}.pdf'.format(plotdir, name))
    plt.clf()


def j(T, C1, C2, W, T0):
    """ This function implements equation (8) in scriptum.
        This is equal to (9) for C2 == 0.
    """
    if C2 != 0:
        integral = np.array([
            quad(lambda x: np.exp(-W / (constants.k * x)), T0, t)[0]
            for t in T
        ])
    else:
        integral = 0
    return (C1
            * np.exp(C2 * integral)
            * np.exp(- W / (constants.k * T))
            )

# Fit and plot for each dataset
for T, I, selection, name in [
        [
            T1,
            I1_cleaned - np.min(I1_cleaned),
            (T1 > 258) & (T1 < 290),
            'set1'
        ],
        [
            T2,
            I2_cleaned - np.min(I2_cleaned),
            (T2 > 250) & (T2 < 267),
            'set2'
        ]]:
    print(name)
    T0 = T[selection][0]
    val, cov = curve_fit(
        lambda x, C1, W: j(x, C1, 0, W, T0),
        T[selection], I[selection], p0=[1, 5e-20]
    )
    errs = np.sqrt(np.diag(cov))
    W = ufloat(val[1], errs[1]) / constants.eV
    with open('{}W_approx_{}.tex'.format(texdir, name), 'w') as f:
        f.write(r'W = \SI{{{:L}}}{{\electronvolt}}'.format(W))
    xs = np.linspace(220, 300, 100)
    plt.ylim(0, 25)
    plt.plot(xs, j(xs, C1=val[0], W=val[1], C2=0, T0=T0), 'r-', label='Fit')
    plt.plot(T[selection], I[selection], 'g.',
             label='Cleaned Data\n(used by fit)')
    plt.plot(T[~selection], I[~selection], 'b.',
             label='Cleaned Data\n(ignored by fit)')
    plt.xlabel(r'$T$ / K')
    plt.ylabel(r'$I_\mathrm{cl}$ / pA')
    plt.legend(loc='best')
    plt.savefig(plotdir + 'fit_approx_{}.pdf'.format(name))
    plt.clf()


def better_fit(T, i_T, Tstar):
    """ Implementation of equation (14) of scriptum.
    """
    integral = np.array(
        [trapz(i_T[(T > t) & (T < Tstar)], T[(T > t) & (T < Tstar)])
         for t in T]
    )
    return np.log(integral / i_T)

for T, I, Tstar, selection, name, b, factor, unit in [
        [
            T1,
            I1_cleaned - np.min(I1_cleaned),
            320,
            (T1 > 260) & (T1 < 310),
            'set1',
            2,
            1e12,
            r'\pico',
        ],
        [
            T2,
            I2_cleaned - np.min(I2_cleaned),
            330,
            (T2 > 250) & (T2 < 300),
            'set2',
            4,
            1e12,
            r'\pico',
        ]]:
    print('better_fit {}'.format(name))
    T = T[selection]
    I = I[selection]
    logstuff = better_fit(T, I, Tstar)
    T = T[np.isfinite(logstuff)]
    logstuff = logstuff[np.isfinite(logstuff)]
    print(T, logstuff)
    var, cov = curve_fit(linear_fit, 1 / T, logstuff)
    errs = np.sqrt(np.diag(cov))
    A = ufloat(var[0], errs[0])
    W = A * constants.k
    Tmax = ufloat(T[np.argmax(I)], 0.1)
    tau0 = ((constants.k * Tmax**2) / (W * b)
            * exp(-W / (constants.k * Tmax)))
    print(tau0)
    with open('{}W_integrated_{}.tex'.format(texdir, name), 'w') as f:
        f.write(r'W = \SI{{{:L}}}{{\electronvolt}}'.format(W / constants.eV))
    with open('{}tau0_integrated_{}.tex'.format(texdir, name), 'w') as f:
        f.write(r'\tau_0 = \SI{{{:L}}}{{{}\second}}'
                .format(tau0 * factor, unit))
    plt.plot(1 / T, logstuff, 'g.', label='Cleaned Data')
    xs = np.linspace(np.min(1/T), np.max(1/T))
    plt.plot(xs, linear_fit(xs, *var), 'r-', label='Fit')
    plt.ticklabel_format(style='sci', scilimits=(0.1, 1000))
    plt.savefig(plotdir + 'integrated-fit-{}.pdf'.format(name))
    ax = plt.gca()
    print([tl.get_text() for tl in ax.get_xticklabels()])
    ax.set_xticklabels([
        r'$\frac{{1}}{{{:.0f}}}$'.format(
            1 / float(tl.get_text())
            / float(ax.get_xaxis().get_offset_text()
                    .get_text().replace('−', '-')))
        if tl.get_text()
        else ''
        for tl in ax.get_xticklabels()
    ])
    plt.xlabel(r'$1/T$ / $1/\mathrm{K}$')
    plt.ylabel(r'$F(T)$')
    plt.legend(loc='best')
    plt.savefig(plotdir + 'integrated-fit-{}.pdf'.format(name))
    plt.clf()
