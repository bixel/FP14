#! /usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import os
from scipy import constants
from scipy.optimize import curve_fit

T = np.arange(-50, 30, 1)
T = constants.C2K(T)
I = np.exp(T)

plotdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '../build/plots/')

plt.plot(T, I)
plt.savefig(plotdir + 'data.pdf')


def j(T, C, W):
    """ equation (9) in script
    """
    return C * np.exp(- W / (constants.k * T))
