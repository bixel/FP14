#! /usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import constants
from scipy.optimize import curve_fit
from scipy.integrate import quad

T = np.arange(-50, 30, 1)
T = constants.C2K(T)
I = np.exp(T)

plotdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '../build/plots/')

plt.plot(T, I)
plt.savefig(plotdir + 'data.pdf')
plt.clf()


def j(T, C1, C2, W, T0):
    """ equation (8) in script
        is equal to (9) for C2 == 0
    """
    integral = np.array([
        quad(lambda x: np.exp(-W / (constants.k * x)), T0, t)[0]
        for t in T
    ])
    return (C1
            * np.exp(C2 * integral[0])
            * np.exp(- W / (constants.k * T))
            )

plt.plot(T, j(T, 1, 1, 2e-18, T[0]))
plt.savefig(plotdir + 'functiontest.pdf')
plt.yscale('log')
plt.clf()
