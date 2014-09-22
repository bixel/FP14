#! /usr/bin/env python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def unbinned_array(binned_array, start=0, end=None,
                   calibration_func=lambda x: x):
    """ Returned an array with 'raw' Data for creating histograms with plt.hist
    """
    raw_data = []
    if end is None:
        end = len(binned_array)
    for x in range(start, end):
        raw_data.extend([calibration_func(x)] * binned_array[x])
    return np.array(raw_data)

data = np.genfromtxt('myons.txt', unpack=True)
unbinned_data = unbinned_array(data)
bins = 50
y, binEdges = np.histogram(unbinned_data, bins=bins)
bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
width = 0.5 * (bincenters[-1] - bincenters[0]) / bins
medStd = np.sqrt(y)
plt.errorbar(bincenters, y, fmt='ro', yerr=medStd, xerr=width)

x_data = np.arange(0, len(data))
decay_function = lambda x, a, b: a * np.exp(b * x)
coeff, var = curve_fit(decay_function, x_data, data, p0=[1, -1e-2])
plt.plot(x_data, decay_function(x_data, coeff[0], coeff[1]))
plt.xlim(0, 400)
plt.show()
