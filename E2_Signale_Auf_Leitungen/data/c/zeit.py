from uncertainties import *
from uncertainties.umath import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

v = 1.98
count = 4

tu = np.array([ufloat(410,5), ufloat(308,5), ufloat(306,5), ufloat(206,5), ufloat(442,5), ufloat(252,5), ufloat(328,5), ufloat(140,5), ufloat(185,5), ufloat(102,5)])
t = np.array([410, 308, 306, 206, 442, 252, 328, 140])

tubar = np.array([ufloat(0,0) for i in range(count)])
tbar = np.array([0.0 for i in range(count)])

lubar = np.array([ufloat(0,0) for i in range(count)])
lbar = np.array([0.0 for i in range(count)])

def calc_t(t1,t2):
	return(abs(t1-t2))

def calc_v(t):
	return(v*t/2)

z = 0
for i in range(count):
	tbar[i] = calc_t(t[z],t[z+1])
	tubar[i] = calc_t(tu[z],tu[z+1])
	lbar[i] = calc_v(tbar[i]*10**(-1))
	lubar[i] = calc_v(tubar[i]*10**(-1))
	z+=2

print(lbar)