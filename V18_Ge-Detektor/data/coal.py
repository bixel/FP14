#! /usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as cfit
from scipy.integrate import simps
from uncertainties import ufloat, umath, unumpy as unp

from hist import unbinned_array

background_data = np.loadtxt("Leermessung.txt", unpack=True)
background_data = unp.uarray(background_data, np.std(background_data))
coal_data = np.loadtxt("kohle.txt", unpack=True)
coal_data = unp.uarray(coal_data, np.std(coal_data))


def cal_energy(x):
    """ calibrate! values from hist.py
        read from json-like file would be nice...
    """
    m = ufloat(0.34521727, 0.00003646)
    b = ufloat(-1.8159, 0.0853)
    return m*x + b


background_xs = cal_energy(np.arange(len(background_data)))
background_xs = unp.uarray([x.n for x in background_xs],
                           [x.s for x in background_xs])
coal_ys = cal_energy(np.arange(len(coal_data)))
coal_ys = unp.uarray([x.n for x in coal_ys],
                     [x.s for x in coal_ys])

plt.savefig('09_background.pdf')
