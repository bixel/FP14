#! /usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import os
from scipy import constants
from scipy.optimize import curve_fit
from scipy.integrate import quad, trapz

T1, I1 = np.genfromtxt('set1.txt', unpack=True)
T2, I2 = np.genfromtxt('set2.txt', unpack=True)
T1 = constants.C2K(T1)
T2 = constants.C2K(T2)
I1_cleaned, I2_cleaned = [], []

min1, max1 = 260, 285
min2, max2 = 250, 267

plotdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '../build/plots/')


# clear data
def linear_fit(T, A, B):
    return A*T + B


def exp_fit(T, A, B, I0):
    return A * np.exp(B * T) + I0


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


for T, I, selection, p0, name in [[T1, I1, (T1 > 280) & (T1 < 294),
                                   None, 'set1'],
                                  [T2, I2, ((T2 > 280) & (T2 < 292)
                                            | (T2 > 245) & (T2 < 260)),
                                   None, 'set2']]:
    fit_function = linear_fit
    var, _ = curve_fit(fit_function, T[selection],
                       I[selection], p0=p0)
    I_cleaned = I - fit_function(T, *var)
    I_cleaned = I_cleaned - np.min(I_cleaned)
    if name == 'set1':
        I1_cleaned = I_cleaned
    else:
        I2_cleaned = I_cleaned
    xs = np.linspace(240, 340)
    plt.plot(xs, fit_function(xs, *var), label='fit')
    plt.plot(T[~selection], I[~selection], 'b.',
             label='Data (ignored for fit)')
    plt.plot(T[selection], I[selection], 'g.', label='Data (used for fit)')
    plt.plot(T, I_cleaned, 'r.', label='Cleaned Data')
    plt.xlim(240, 320)
    plt.xlabel(r'$T$ / K')
    plt.ylabel(r'$I$ / pA')
    plt.legend(loc='best')
    if p0:
        plt.plot(xs, fit_function(xs, *p0))
    plt.savefig('{}cleaned_data-{}.pdf'.format(plotdir, name))
    plt.clf()


def j(T, C1, C2, W, T0):
    """ This function implements equation (8) in scriptum.
        This is equal to (9) for C2 == 0.
    """
    integral = np.array([
        quad(lambda x: np.exp(-W / (constants.k * x)), T0, t)[0]
        for t in T
    ])
    return (C1
            * np.exp(C2 * integral[0])
            * np.exp(- W / (constants.k * T))
            )

# Fit and plot for each dataset
for T, I, selection, name in [[T1, I1_cleaned,
                              (T1 > 260) & (T1 < 280), 'set1'],
                              [T2, I2_cleaned,
                              (T2 > 250) & (T2 < 267), 'set2']]:
    T0 = T[selection][0]
    val, cov = curve_fit(
        lambda x, C1, W: j(x, C1, 0, W, T0),
        T[selection], I[selection], p0=[1, 5e-20]
    )
    errs = np.sqrt(np.diag(cov))
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

for T, I, Tstar, selection, name in [[T1, I1_cleaned, 280,
                                     (T1 > 250) & (T1 < 275), 'set1'],
                                     [T2, I2_cleaned, 295,
                                     (T2 > 250) & (T2 < 284), 'set2']]:
    T = T[selection]
    I = I[selection]
    logstuff = better_fit(T, I, Tstar)
    T = T[np.isfinite(logstuff)]
    logstuff = logstuff[np.isfinite(logstuff)]
    print(logstuff)
    var, cov = curve_fit(linear_fit, 1 / T, logstuff)
    errs = np.sqrt(np.diag(cov))
    print(var, errs)
    plt.plot(1 / T, logstuff, 'g.', label='Cleaned Data')
    xs = np.linspace(np.min(1/T), np.max(1/T))
    plt.plot(xs, linear_fit(xs, *var), 'r-', label='Fit')
    plt.ticklabel_format(style='sci', scilimits=(0.1, 1000))
    plt.xlabel(r'$1/T$ / $1/\mathrm{K}$')
    plt.ylabel(r'$F(T)$')
    plt.legend(loc='best')
    plt.savefig(plotdir + 'integrated-fit-{}.pdf'.format(name))
    plt.clf()
