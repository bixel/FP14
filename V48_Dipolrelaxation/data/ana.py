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
    print(var)
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
for T, I, min_T, max_T, name in [[T1, I1_cleaned, min1, max1, 'set1'],
                                 [T2, I2_cleaned, min2, max2, 'set2']]:
    T_fit_values = T[(T > min_T) & (T < max_T)]
    I_fit_values = I[(T > min_T) & (T < max_T)]
    print(T_fit_values, I_fit_values)

    val, cov = curve_fit(
        lambda x, C1, W: j(x, C1, 0, W, T[T > min1][0]),
        T_fit_values, I_fit_values, p0=[1, 5e-20]
    )
    errs = np.sqrt(np.diag(cov))
    print(val, errs)

    xs = np.linspace(min_T, max_T, 100)
    # mark_fit_area((200, min_T), (0, 25), alpha=0.2, edgecolor='none')
    # mark_fit_area((max_T, 340), (0, 25), alpha=0.2, edgecolor='none')
    plt.plot(T, I, '+')
    plt.plot(xs, j(xs, C1=val[0], W=val[1], C2=0, T0=250))
    plt.savefig(plotdir + 'fit_approx_{}.pdf'.format(name))
    plt.clf()


def better_fit(T, i_T, Tstar):
    """ Implementation of equation (14) of scriptum.
    """
    integral = np.array(
        [trapz(i_T[(T > t) & (T < Tstar)], T[(T > t) & (T < Tstar)])
         for t in T]
    )
    print(integral)
    return np.log(integral / i_T)

for T, I, min_T, name in [[T1, I1_cleaned, 310, 'set1'],
                          [T2, I2_cleaned, 320, 'set2']]:
    logstuff = better_fit(T, I, min_T)
    plt.plot(1 / T, logstuff, '+')
    plt.savefig(plotdir + 'integrated-fit-{}.pdf'.format(name))
    plt.clf()
