#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

start, stop = 1900, 1950
caesium = np.genfromtxt('Caesium.txt', unpack=True)[start:stop]

plt.bar(np.arange(start, stop), caesium, width=1.0, linewidth=0)
plt.axhline(0.5 * max(caesium), linestyle='--', color='r')
plt.axhline(0.1 * max(caesium), linestyle='--', color='r')
plt.savefig('peak.pdf')

a = ufloat(len(caesium[caesium >= 0.5 * max(caesium)]), 1)
b = ufloat(len(caesium[caesium >= 0.1 * max(caesium)]), 1)

print(b/a)
