#! /usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np


def gauss(x, mu=0, sigma=0):
    return 1 / (sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x - mu)/sigma)**2)


def gamma(x, t=0, s=1):
    return s / (s**2 + (x-t)**2) / np.pi


xs = np.linspace(3, 7, 100000)
ys = gauss(xs, mu=5, sigma=0.5)
plt.plot(xs, ys)
for mu in np.arange(np.min(xs), np.max(xs), 0.1):
    inner_ys = 0.003 * gamma(xs, t=mu, s=0.001) * ys
    plt.plot(xs, inner_ys, 'r-')
plt.xlabel(r'$f$')
plt.ylabel(r'$I$')
plt.xticks([])
plt.yticks([])
plt.legend(['Gaußprofil', 'Spektrum'], loc='best')
plt.savefig('build/gauß.pdf')
