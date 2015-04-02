# coding: utf-8
import numpy as np
from uncertainties import ufloat
from matplotlib import pyplot as plt


# Teil c)
# Momentanspannung
offset = ufloat(16, 0.1)
U1 = ufloat(83.5, 0.1) + offset
U2 = ufloat(34, 0.1) + offset

m1 = (U1 - U2) / (U1 + U2)
print(m1)
with open('build/m1.tex', 'w') as f:
    f.write(r'\num{{{:L}}}'.format(m1))

# Spektrum
UT = ufloat(4.7, 0.1)
U1 = ufloat(1.57, 0.01)
U2 = ufloat(1.56, 0.01)

m2 = (U1 + U2) / UT
print(m2)
with open('build/m2.tex', 'w') as f:
    f.write(r'\num{{{:L}}}'.format(m2))

# Teil e)
# Abhängigkeit phase-Spannung
T, U = np.genfromtxt('am-demodulation-e.txt', unpack=True)
plt.plot(T, U, 'r+')
plt.xlabel(r'$T/\mathrm{ns}$')
plt.ylabel(r'$U/\mathrm{mV}$')
plt.savefig('build/demodulation-cosinus.pdf')
plt.clf()
