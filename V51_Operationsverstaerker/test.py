import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit

p = 2
z = 430
L = 100 

x = 10
y = 20

s = np.sqrt(x**2+y**2)
print(s)
print(x**2+y**2)

I = 1/2 * p**2 /430 * 10**( - L / 10) * s**2 / 4

print(I)

ro = 1e3
g = 10
p = 2

x = p/(10e3)
print(x)