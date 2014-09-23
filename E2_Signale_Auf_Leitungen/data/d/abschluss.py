from uncertainties import *
from uncertainties.umath import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

Xa, Uar = np.loadtxt("ALL0004/F0004CH1.CSV", comments="%" ,unpack=True)
Xb, Ubr = np.loadtxt("ALL0005/F0005CH1.CSV", comments="%" ,unpack=True)
Xc, Ucr = np.loadtxt("ALL0007/F0007CH1.CSV", comments="%" ,unpack=True)

Z = 50

def sum_it(x,y,z):
	sum1 = 0
	for i in range(x):
		sum1 += z[i+y]
	return(sum1/x)

Ua = np.array([ufloat(0,0) for i in range(len(Xa))])
Ub = np.array([ufloat(0,0) for i in range(len(Xa))])
Uc = np.array([ufloat(0,0) for i in range(len(Xa))])

for i in range(len(Xa)):
	Ua[i] = ufloat(Uar[i], 0.5)
	Ub[i] = ufloat(Ubr[i], 0.5)
	Uc[i] = ufloat(Ucr[i], 0.5)

U0_a = sum_it(100,0,Ua)
U0_b = sum_it(100,0,Ub)
U0_c = sum_it(100,0,Uc)
U1_a = sum_it(656-600,600,Ua)
U1_b = sum_it(350-180,180,Ub)
U1_c = sum_it(1329-594,594,Uc)
U2_a = sum_it(100,2482-100,Ua)
U2_b = sum_it(100,2482-100,Ub)
U2_c = sum_it(100,2482-100,Uc)

U1_a -= U0_a
U2_a -= U0_a
U1_b -= U0_b
U2_b -= U0_b
U1_c -= U0_c
U2_c -= U0_c

print("U0_a:" + str(U0_a))
print("U0_b:" + str(U0_b))
print("U0_c:" + str(U0_c))
print("U1_a:" + str(U1_a))
print("U2_a:" + str(U2_a))
print("U1_b:" + str(U1_b))
print("U2_b:" + str(U2_b))
print("U1_c:" + str(U1_c))
print("U2_c:" + str(U2_c))

gamma_a = U1_a/U2_a - 1
gamma_b = U2_b/U1_b - 1
gamma_c = U1_c/U2_c - 1
print("g_a:" + str(gamma_a))
print("g_b:" + str(gamma_b))
print("g_c:" + str(gamma_c))
R_a = (Z+gamma_a*Z)/(1-gamma_a)
R_b = (Z+gamma_b*Z)/(1-gamma_b)
R_c = (Z+gamma_c*Z)/(1-gamma_c)
print("R_a:" + str(R_a))
print("R_b:" + str(R_b))
print("R_c:" + str(R_c))

start = 1000
end = 2200

Ua_exp = np.array([ufloat(0,0) for i in range(end-start)])
Ua_expr = np.linspace(0, end-start, end-start)
Ua_r = np.array([0.0 for i in range(end-start)])

for i in range(end-start):
	Ua_exp[i] = log(abs(Ua[i+start]-U2_a))
	Ua_r[i] = Ua_exp[i].nominal_value
	Ua_expr[i] = Xa[i+start]

start = 500
end = 1450

Ub_exp = np.array([ufloat(0,0) for i in range(end-start)])
Ub_expr = np.linspace(0,end-start,end-start)
Ub_r = np.array([0.0 for i in range(end-start)])

for i in range(end-start):
	Ub_exp[i] = log(abs(Ub[i+start]-U2_b))
	Ub_r[i] = Ub_exp[i].nominal_value
	Ub_expr[i] = Xb[i+start]

start = 1600
end = 2200

Uc_exp = np.array([ufloat(0,0) for i in range(end-start)])
Uc_expr = np.linspace(0, end-start, end-start)
Uc_r = np.array([0.0 for i in range(end-start)])

for i in range(end-start):
	Uc_exp[i] = log(abs(Uc[i+start]-U2_c))
	Uc_r[i] = Uc_exp[i].nominal_value
	Uc_expr[i] = Xc[i+start]

val1 = stats.linregress(Ua_expr, Ua_r)
val2 = stats.linregress(Ub_expr, Ub_r)
val3 = stats.linregress(Uc_expr, Uc_r)

m_a = ufloat(val1[0],val1[4])
m_b = ufloat(val2[0],val2[4])
m_c = ufloat(val3[0],val3[4])

print("m_a:" +str(m_a))
print("m_b:" +str(m_b))
print("m_c:" +str(m_c))

print("b_a:" +str(val1[1]))
print("b_b:" +str(val2[1]))
print("b_c:" +str(val3[1]))

L_a = -(Z+R_a)/m_a
L_b = -(Z+R_b)/m_b
L_c = -(Z+R_c)/m_c

print("L_a:" + str(L_a))
print("L_b:" + str(L_b))
print("L_c:" + str(L_c))

C_a = -1/(m_a*Z)
C_b = -1/(m_b*Z)
C_c = -1/(m_c*Z)

print("C_a:" + str(C_a))
print("C_b:" + str(C_b))
print("C_c:" + str(C_c))

plt.plot(Ua_expr, val1[0]*Ua_expr+val1[1], "r-", label="Regression geschlossen")
plt.plot(Ub_expr, val2[0]*Ub_expr+val2[1], "b-", label="Regression offen")
plt.plot(Uc_expr, val3[0]*Uc_expr+val3[1], "g-", label="Regression offen")

plt.plot(Ua_expr,Ua_r,"r.", label="Messpunkte offen")
plt.plot(Ub_expr,Ub_r,"b.", label="Messpunkte geschlossen")
plt.plot(Uc_expr,Uc_r,"g.", label="Messpunkte geschlossen")
plt.xlabel("t[s]")
plt.ylabel("ln(U[mV]/1[mV])")

d = 0.9*10**(-3)
D = 2.95*10**(-3)

epsilon_a = ( sqrt(L_a/C_a) / (log(D/d) * 60))**(-2)
epsilon_b = ( sqrt(L_b/C_b) / (log(D/d) * 60))**(-2)
epsilon_c = ( sqrt(L_c/C_c) / (log(D/d) * 60))**(-2)
print("e_a:" + str(epsilon_a))
print("e_b:" + str(epsilon_b))
print("e_c:" + str(epsilon_c))

plt.show()