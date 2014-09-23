from uncertainties import *
from uncertainties.umath import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

count_o0 = 100
count_o1 = 1682-823
count_o2 = 100
count_g0 = 100
count_g1 = 1144-844
count_g2 = 100
Z = 50

def sum_it(x,y,z):
	sum1 = 0
	for i in range(x):
		sum1 += z[i+y]
	return(sum1/x)

Xg, Ugr = np.loadtxt("ALL0002/F0002CH1.CSV", comments="%" ,unpack=True)
Xo, Uor = np.loadtxt("ALL0003/F0003CH1.CSV", comments="%" ,unpack=True)

Uo = np.array([ufloat(0,0) for i in range(len(Xg))])
Ug = np.array([ufloat(0,0) for i in range(len(Xg))])

for i in range(len(Xg)):
	Uo[i] = ufloat(Uor[i], 0.5)
	Ug[i] = ufloat(Ugr[i], 0.5)

U0_g = sum_it(count_g0,0,Ug)
U1_g = sum_it(count_g1,844,Ug)
U2_g = sum_it(count_g2,2482-100,Ug)
U0_o = sum_it(count_o0,0,Uo)
U1_o = sum_it(count_o1,823,Uo)
U2_o = sum_it(count_o2,2482-100,Uo)

U1_o -= U0_o
U2_o -= U0_o
U1_g -= U0_g
U2_g -= U0_g
print("U0_o:"+str(U0_o))
print("U1_o:" + str(U1_o))
print("U2_o:" + str(U2_o))
print("U0_g:"+str(U0_g))
print("U1_g:" + str(U1_g))
print("U2_g:" + str(U2_g))

gamma_o = U1_o/U2_o - 1
gamma_g = U2_g/U1_g - 1
print("g_o:" + str(gamma_o))
print("g_g:" + str(gamma_g))
R_o = (Z+gamma_o*Z)/(1-gamma_o)
R_g = (Z+gamma_g*Z)/(1-gamma_g)
print("R_o:" + str(R_o))
print("R_g:" + str(R_g))

start = 1676
end = 1767

Ug_exp = np.array([ufloat(0,0) for i in range(end-start)])
Ug_expr = np.linspace(0,end-start,end-start)
Ug_r = np.array([0.0 for i in range(end-start)])

for i in range(end-start):
	Ug_exp[i] = log(abs(Ug[i+start]-U2_g))
	Ug_r[i] = Ug_exp[i].nominal_value
	Ug_expr[i] = Xg[i+start]

start = 1200
end = 1400

Uo_exp = np.array([ufloat(0,0) for i in range(end-start)])
Uo_expr = np.linspace(0, end-start, end-start)
Uo_r = np.array([0.0 for i in range(end-start)])

for i in range(end-start):
	Uo_exp[i] = log(abs(Uo[i+start]-U2_o))
	Uo_r[i] = Uo_exp[i].nominal_value
	Uo_expr[i] = Xo[i+start]

val1 = stats.linregress(Ug_expr, Ug_r)
val2 = stats.linregress(Uo_expr, Uo_r)

m_g = ufloat(val1[0],val1[4])
m_o = ufloat(val2[0],val2[4])

print("m_g:" +str(m_g))
print("m_o:" +str(m_o))

print("b_g:" +str(val1[1]))
print("b_o:" +str(val2[1]))

L_g = -(Z+R_g)/m_g
L_o = -(Z+R_o)/m_o

print("L_g:" + str(L_g))
print("L_o:" + str(L_o))

C_g = -1/(m_g*Z)
C_o = -1/(m_o*Z)

print("C_g:" + str(C_g))
print("C_o:" + str(C_o))

plt.plot(Ug_expr, val1[0]*Ug_expr+val1[1], "g-", label="Regression geschlossen")
plt.plot(Uo_expr, val2[0]*Uo_expr+val2[1], "k-", label="Regression offen")

plt.plot(Uo_expr,Uo_r,"r.", label="Messpunkte offen")
plt.plot(Ug_expr,Ug_r,"b.", label="Messpunkte geschlossen")
plt.xlabel("t[s]")
plt.ylabel("ln(U[mV]/1[mV])")
plt.legend()
plt.savefig("Regression.pdf")
########### Smith Diagramm

f = 1000 #kHz
e = 2.25 #epsilon_r
c_o = 3*10**(8)
def goforZ(a,b,c):
	return(complex(a,2*np.pi*b*c))

def goforZ2(b,c):
	return(complex(0,-1/(2*np.pi*b*c)))

def gamma(a,b):
	return((a-b)/(a+b))

def length(a,b):
	return(a*b/(4*np.pi))

Z_o = goforZ(R_o.nominal_value,f,L_o.nominal_value)
Z_g = goforZ(R_g.nominal_value,f,L_g.nominal_value)
print("Z_o:"+str(Z_o))
print("Z_g:"+str(Z_g))
gamma_gd = gamma(Z_g,Z)
gamma_od = gamma(Z_o,Z)
print("gamma_od:"+str(gamma_od))
print("gamma_gd:"+str(gamma_gd))
theta_o = np.arccos((gamma_od.real*gamma_o.nominal_value)/(abs(gamma_od)*abs(gamma_o.nominal_value)))
theta_g = np.arccos((gamma_gd.real*gamma_g.nominal_value)/(abs(gamma_gd)*abs(gamma_g.nominal_value)))
print("theta_o:"+str(theta_o))
print("theta_g:"+str(theta_g))
lamb = c_o/(f*sqrt(e))
print("lambda:"+str(lamb))
l_o = length(theta_o,lamb)
l_g = length(theta_g,lamb)
print("leng_o:"+str(l_o))
print("leng_g:"+str(l_g))

Z_o = goforZ2(f,C_o.nominal_value)
Z_g = goforZ2(f,C_g.nominal_value)
print("Z_o:"+str(Z_o))
print("Z_g:"+str(Z_g))
gamma_gd = gamma(Z_g,Z)
gamma_od = gamma(Z_o,Z)
print("gamma_od:"+str(gamma_od))
print("gamma_gd:"+str(gamma_gd))
theta_o = np.arccos((gamma_od.real*gamma_o.nominal_value)/(abs(gamma_od)*abs(gamma_o.nominal_value)))
theta_g = np.arccos((gamma_gd.real*gamma_g.nominal_value)/(abs(gamma_gd)*abs(gamma_g.nominal_value)))
print("theta_o:"+str(theta_o))
print("theta_g:"+str(theta_g))
lamb = c_o/(f*sqrt(e))
print("lambda:"+str(lamb))
l_o = length(theta_o,lamb)
l_g = length(theta_g,lamb)
print("leng_o:"+str(l_o))
print("leng_g:"+str(l_g))
