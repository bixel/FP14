from uncertainties import *
from uncertainties.umath import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

def sum_it(x,y,z):
	sum1 = 0
	for i in range(x):
		sum1 += z[i+y]
	return(sum1/x)

Xa, Uar = np.loadtxt("ALL0008/F0008CH1.CSV", comments="%" ,unpack=True)
Xb, Ubr = np.loadtxt("ALL0009/F0009CH1.CSV", comments="%" ,unpack=True)
Xc, Ucr = np.loadtxt("ALL0010/F0010CH1.CSV", comments="%" ,unpack=True)

Ua = np.array([ufloat(0,0) for i in range(len(Xa))])
Ub = np.array([ufloat(0,0) for i in range(len(Xa))])
Uc = np.array([ufloat(0,0) for i in range(len(Xa))])

for i in range(len(Xa)):
	Ua[i] = ufloat(Uar[i], 0.5)
	Ub[i] = ufloat(Ubr[i], 0.5)
	Uc[i] = ufloat(Ucr[i], 0.5)

U0_a = sum_it(100,0,Ua)
U1_a = sum_it(750-355,355,Ua)
U2_a = sum_it(1050-875,875,Ua)
U3_a = sum_it(1830-1520,1520,Ua)
U4_a = sum_it(100, 2300, Ua)
U0_b = sum_it(100,0,Ub)
U1_b = sum_it(600-252,252,Ub)
U2_b = sum_it(1100-800,800,Ub)
U3_b = sum_it(1700-1400,1400,Ub)
U0_c = sum_it(100,0,Uc)
U1_c = sum_it(630-260,260,Uc)
U2_c = sum_it(1000-800,800,Uc)
U3_c = sum_it(1800-1400,1400,Uc)

U1_a -= U0_a
U2_a -= U0_a
U3_a -= U0_a
U4_a -= U0_a
U1_b -= U0_b
U2_b -= U0_b
U3_b -= U0_b
U1_c -= U0_c
U2_c -= U0_c
U3_c -= U0_c

print("U0_a:" + str(U0_a))
print("U1_a:" + str(U1_a))
print("U2_a:" + str(U2_a))
print("U3_a:" + str(U3_a))
print("U4_a:" + str(U4_a))

dU1a = U2_a - U1_a
dU2a = U3_a - U2_a
dU3a = U4_a - U3_a

dU1b = U1_b - U0_b
dU2b = U2_b - U1_b
dU3b = U3_b - U2_b

dU1c = U1_c - U0_c
dU2c = U2_c - U1_c
dU3c = U3_c - U2_c

print("dU1a:" + str(dU1a))
print("dU2a:" + str(dU2a))
print("dU3a:" + str(dU3a))


gamma1_a = (dU1a / U1_a)
gamma2_a = (dU3a / dU2a) + dU2a / (U1_a * (1-gamma1_a))
gamma3_a = dU3a / (dU2a * gamma2_a)

gamma1_b = dU1b / U0_b
gamma2_b = dU3b / dU2b + dU2b / (U0_b * (1-gamma1_b))
gamma3_b = dU3b / (dU2b * gamma2_b)

gamma1_c = dU1c / U0_c
gamma2_c = dU3c / dU2c + dU2c / (U0_c * (1-gamma1_c))
gamma3_c = dU3c / (dU2c * gamma2_c)

print("gamma1_a:" + str(gamma1_a))
print("gamma2_a:" + str(gamma2_a))
print("gamma3_a:" + str(gamma3_a))

