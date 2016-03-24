#! /usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def gauss(x, nu):
    return np.exp(-(x - nu) ** 2)

xs = np.linspace(-4, 12, 500)
plt.plot(xs, 1 - (gauss(xs, 0) + 0.7 * gauss(xs, 8)))
plt.xlabel(r'$B_\mathrm{m}$')
plt.ylabel(r'Transparenz')
plt.ylim(0, 1.1)
plt.xticks([0])
plt.yticks([1])
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['top'].set_position('zero')
ax.spines['left'].set_position('zero')
ax.xaxis.labelpad = 0.05
ax.yaxis.labelpad = 0.05
plt.tight_layout()
plt.savefig('build/plots/transmission.pdf')
plt.clf()
