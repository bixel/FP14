#! /usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt

theta, I = np.genfromtxt('data/polarisation.txt', unpack=True)

# close the circle
theta = np.append(theta, 360)
I = np.append(I, I[0])

plt.polar(theta * (np.pi / 180), I, 'bo--')
plt.savefig('build/polarisation_polar.pdf')
plt.clf()

theta = theta[:-1]
I = I[:-1]
plt.errorbar(theta, I, yerr=0.05*I, fmt='bo')
plt.xlabel(r'$\theta/^\circ$')
plt.ylabel(r'$I/\mathrm{mA}$')
plt.xlim(-10, 360)
plt.savefig('build/polarisation_lin.pdf')
plt.clf()

