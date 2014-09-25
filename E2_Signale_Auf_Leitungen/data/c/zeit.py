from uncertainties import *
from uncertainties.umath import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

v = 1.98*10**8
count = 6

tu = np.array([ufloat(00.000000025400,5*10**(-9)), ufloat(-00.000000085000,5*10**(-9)), ufloat(00.000000125100,5*10**(-9)), ufloat(00.000000000100,5*10**(-9)), ufloat(-00.000000000000,5*10**(-9)), ufloat(00.000001081000,5*10**(-9)), ufloat(00.000000118000,5*10**(-9)), ufloat(-00.000000810000,5*10**(-9)), ufloat( 00.000000019000,5*10**(-9)), ufloat(00.000000209000,5*10**(-9)), ufloat( -00.000000012600,5*10**(-9)), ufloat(00.000000195600,5*10**(-9))])
t = np.array([00.000000025400, -00.000000085000, 00.000000125100, 00.000000000100, -00.000000000000, 00.000001081000, 00.000000118000, -00.000000810000, 00.000000019000, 00.000000209000, -00.000000012600, 00.000000195600])

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
	lbar[i] = calc_v(tbar[i])
	lubar[i] = calc_v(tubar[i])
	z+=2

print(tubar)
print(lubar)