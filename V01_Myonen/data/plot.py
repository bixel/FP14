#! /usr/bin/evn python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def unbinned_array(binned_array, start=0, end=None, calibration_func=lambda x: x):
    """ Returned an array with 'raw' Data for creating histograms with plt.hist
    """
    raw_data = []
    if end == None:
        end = len(binned_array) - 1
    for x in range(start,end):
        raw_data.extend([calibration_func(x)] * binned_array[x])
    return raw_data

data = np.genfromtxt('myons.txt', unpack=True)
unbinned_data = unbinned_array(data)
plt.hist(unbinned_data, bins=200)
plt.show()
