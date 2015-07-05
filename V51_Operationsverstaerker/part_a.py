import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit
from codecs import open
from textable import table

R_N_1 = 1e6
R_N_2 = 120e3
R_N_3 = 4.7e3
R_N_4 = 1e6

R_1_1 = 100
R_1_2 = 100
R_1_3 = 100
R_1_4 = 4.7e3

R_in = 50
U_ein = 15e-3

R_1_1 += R_in
R_1_2 += R_in
R_1_3 += R_in
R_1_4 += R_in

print(R_1_1)
print(R_1_2)
print(R_1_3)
print(R_1_4)

# Import Data
nu_1, U_1, dU_1 = np.genfromtxt("data/data_a_1.txt",unpack=True)
nu_2, U_2, dU_2 = np.genfromtxt("data/data_a_2.txt",unpack=True)
nu_3, U_3, dU_3 = np.genfromtxt("data/data_a_3.txt",unpack=True)
nu_4, U_4, dU_4 = np.genfromtxt("data/data_a_4.txt",unpack=True)

with open('build/a_data_1.tex', 'w') as f:
	f.write(table([r'$\nu / \si{\hertz}$', r'$U / \si{\volt}$'],
		[nu_1, unp.uarray(U_1, dU_1)]
		))
with open('build/a_data_2.tex', 'w') as f:
	f.write(table([r'$\nu / \si{\hertz}$', r'$U / \si{\volt}$'],
		[nu_2, unp.uarray(U_2, dU_2)]
		))
with open('build/a_data_3.tex', 'w') as f:
	f.write(table([r'$\nu / \si{\hertz}$', r'$U / \si{\volt}$'],
		[nu_3, unp.uarray(U_3, dU_3)]
		))
with open('build/a_data_4.tex', 'w') as f:
	f.write(table([r'$\nu / \si{\hertz}$', r'$U / \si{\volt}$'],
		[nu_4, unp.uarray(U_4, dU_4)]
		))

V_1 = unp.uarray(U_1,dU_1) / U_ein
V_2 = unp.uarray(U_2,dU_2) / U_ein
V_3 = unp.uarray(U_3,dU_3) / U_ein
V_4 = unp.uarray(U_4,dU_4) / U_ein

nu_1_log, V_1_log = np.log(nu_1)+0, unp.log(V_1)+0
nu_2_log, V_2_log = np.log(nu_2)+0, unp.log(V_2)+0
nu_3_log, V_3_log = np.log(nu_3)+0, unp.log(V_3)+0
nu_4_log, V_4_log = np.log(nu_4)+0, unp.log(V_4)+0

# Fit Data
def func(x,a,b):
	return a*x+b

popt_1, pcov_1 = curve_fit(func,nu_1_log,unp.nominal_values(V_1_log))
popt_2, pcov_2 = curve_fit(func,nu_2_log[6:14],unp.nominal_values(V_2_log[6:14]))
popt_3, pcov_3 = curve_fit(func,nu_3_log[12:18],unp.nominal_values(V_3_log[12:18]))
popt_4, pcov_4 = curve_fit(func,nu_4_log[11:18],unp.nominal_values(V_4_log[11:18]))

m = np.array([ufloat(popt_1[0],pcov_1[0][0]),
				ufloat(popt_2[0],pcov_2[0][0]),
				ufloat(popt_3[0],pcov_3[0][0]),
				ufloat(popt_4[0],pcov_4[0][0])])

b = np.array([ufloat(popt_1[1],pcov_1[1][1]),
				ufloat(popt_2[1],pcov_2[1][1]),
				ufloat(popt_3[1],pcov_3[1][1]),
				ufloat(popt_4[1],pcov_4[1][1])])
print(m)
print(b)

# Calcs

def verstaerkung(v,r_1,r_N):
	return ( - r_1 / r_N + 1 / v )**-1

V_grenz_1 = V_1[0] / np.sqrt(2)
V_grenz_2 = V_2[0] / np.sqrt(2)
V_grenz_3 = V_3[0] / np.sqrt(2)
V_grenz_4 = V_4[0] / np.sqrt(2)

V_grenz_1_log = unp.log(V_grenz_1) + 0
V_grenz_2_log = unp.log(V_grenz_2) + 0
V_grenz_3_log = unp.log(V_grenz_3) + 0
V_grenz_4_log = unp.log(V_grenz_4) + 0

nu_grenz_1_log = ( V_grenz_1_log - popt_1[1] ) / popt_1[0]
nu_grenz_2_log = ( V_grenz_2_log - popt_2[1] ) / popt_2[0]
nu_grenz_3_log = ( V_grenz_3_log - popt_3[1] ) / popt_3[0]
nu_grenz_4_log = ( V_grenz_4_log - popt_4[1] ) / popt_4[0]

nu_grenz_1 = unp.exp(nu_grenz_1_log) + 0
nu_grenz_2 = unp.exp(nu_grenz_2_log) + 0 
nu_grenz_3 = unp.exp(nu_grenz_3_log) + 0
nu_grenz_4 = unp.exp(nu_grenz_4_log) + 0

print("nu1 = " + str(nu_grenz_1))
print("nu2 = " + str(nu_grenz_2))
print("nu3 = " + str(nu_grenz_3))
print("nu4 = " + str(nu_grenz_4))

with open('build/a_nu_1.tex', 'w') as f:
	f.write(r'\SI{{{:L}}}{{\hertz}}'.format(nu_grenz_1))
with open('build/a_nu_2.tex', 'w') as f:
	f.write(r'\SI{{{:L}}}{{\kilo\hertz}}'.format(nu_grenz_2 / 1e3 ))
with open('build/a_nu_3.tex', 'w') as f:
	f.write(r'\SI{{{:L}}}{{\kilo\hertz}}'.format(nu_grenz_3 / 1e3 ))
with open('build/a_nu_4.tex', 'w') as f:
	f.write(r'\SI{{{:L}}}{{\kilo\hertz}}'.format(nu_grenz_4 / 1e3 ))   

print("nu1*V1 = " + str(nu_grenz_1 * V_1[0]))
print("nu2*V2 = " + str(nu_grenz_2 * V_2[0]))
print("nu3*V3 = " + str(nu_grenz_3 * V_3[0]))
print("nu4*V4 = " + str(nu_grenz_4 * V_4[0]))

with open('build/a_nu_1_V_1.tex', 'w') as f:
	f.write(r'\SI{{{:L}}}{{\kilo\hertz}}'.format(nu_grenz_1 * V_1[0] / 1e3 ))
with open('build/a_nu_2_V_2.tex', 'w') as f:
	f.write(r'\SI{{{:L}}}{{\kilo\hertz}}'.format(nu_grenz_2 * V_2[0] / 1e3 ))
with open('build/a_nu_3_V_3.tex', 'w') as f:
	f.write(r'\SI{{{:L}}}{{\kilo\hertz}}'.format(nu_grenz_3 * V_3[0] / 1e3 ))
with open('build/a_nu_4_V_4.tex', 'w') as f:
	f.write(r'\SI{{{:L}}}{{\kilo\hertz}}'.format(nu_grenz_4 * V_4[0] / 1e3 ))

print('V_1 = ' + str( verstaerkung(V_1[0],R_1_1,R_N_1) ))
print('V_2 = ' + str( verstaerkung(V_2[0],R_1_2,R_N_2) ))
print('V_3 = ' + str( verstaerkung(V_3[0],R_1_3,R_N_3) ))
print('V_4 = ' + str( verstaerkung(V_4[0],R_1_4,R_N_4) ))

with open('build/a_V_1.tex', 'w') as f:
	f.write(r'\num{{{:L}}}'.format( verstaerkung(V_1[0],R_1_1,R_N_1) ))
with open('build/a_V_2.tex', 'w') as f:
	f.write(r'\num{{{:L}}}'.format( verstaerkung(V_2[0],R_1_2,R_N_2) ))
with open('build/a_V_3.tex', 'w') as f:
	f.write(r'\num{{{:L}}}'.format( verstaerkung(V_3[0],R_1_3,R_N_3) ))
with open('build/a_V_4.tex', 'w') as f:
	f.write(r'\num{{{:L}}}'.format( verstaerkung(V_4[0],R_1_4,R_N_4) ))


print('V_1 = ' + str(V_1[0]) + ' dR = ' + str(R_N_1 / R_1_1))
print('V_2 = ' + str(V_2[0]) + ' dR = ' + str(R_N_2 / R_1_2))
print('V_3 = ' + str(V_3[0]) + ' dR = ' + str(R_N_3 / R_1_3))
print('V_4 = ' + str(V_4[0]) + ' dR = ' + str(R_N_4 / R_1_4))

with open('build/a_dR_1.tex', 'w') as f:
	f.write(r'\num{{{:0.4f}}}'.format( R_N_1 / R_1_1 ))
with open('build/a_dR_2.tex', 'w') as f:
	f.write(r'\num{{{:0.4f}}}'.format( R_N_2 / R_1_2 ))
with open('build/a_dR_3.tex', 'w') as f:
	f.write(r'\num{{{:0.4f}}}'.format( R_N_3 / R_1_3 ))
with open('build/a_dR_4.tex', 'w') as f:
	f.write(r'\num{{{:0.4f}}}'.format( R_N_4 / R_1_4 ))


# Plot Data
x_1 = np.linspace(nu_1_log[0]-1,nu_1_log[len(nu_1_log)-1]+1,10**5)
x_2 = np.linspace(nu_2_log[6]-1,nu_2_log[14]+1,10**5)
x_3 = np.linspace(nu_3_log[12]-1,nu_3_log[18]+1,10**5)
x_4 = np.linspace(nu_4_log[11]-1,nu_4_log[18]+1,10**5)

x_1_full = np.linspace(nu_1_log[0],nu_1_log[len(nu_1_log)-1],10**5)
x_2_full = np.linspace(nu_2_log[0],nu_2_log[len(nu_2_log)-1],10**5)
x_3_full = np.linspace(nu_3_log[0],nu_3_log[len(nu_3_log)-1],10**5)
x_4_full = np.linspace(nu_4_log[0],nu_4_log[len(nu_4_log)-1],10**5)

plt.title(r'Frequenzgang eines gekoppelten Verstärkers mit:'+ '\n' 
	+ r'$R_N = 1\,\mathrm{M}\Omega;\, R_1 = 100\,\Omega$')
plt.plot(nu_1_log, unp.nominal_values(V_1_log),"xb",label=r'Messpunkte')
plt.plot(x_1,func(x_1,*popt_1),'-k', label=r'Ausgleichsgerade $f_1(x)$')
plt.legend(loc='best')
plt.grid()
plt.xlabel(r'$\log(\nu/\mathrm{Hz})$')
plt.ylabel(r'$\log(\overline{V})$')
plt.savefig('build/a_plot_1.pdf')
plt.clf()

plt.title(r'Frequenzgang eines gekoppelten Verstärkers mit:'+ '\n' 
	+ r'$R_N = 120\,\mathrm{k}\Omega;\, R_1 = 100\,\Omega$')
plt.plot(nu_2_log, unp.nominal_values(V_2_log),"xb",label=r'Messpunkte')
plt.plot(x_2,func(x_2,*popt_2),'-k', label=r'Ausgleichsgerade $f_2(x)$')
plt.axhline(y=unp.nominal_values(V_grenz_2_log),color='g',linestyle='--', 
	label=r'$\log(\overline{V}/\sqrt{2})$')
plt.axvline(x=unp.nominal_values(nu_grenz_2_log),color='r',linestyle='--',
	label=r'$\log(\nu_g/\mathrm{Hz})$')
plt.legend(loc='best')
plt.grid()
plt.xlabel(r'$\log(\nu/\mathrm{Hz})$')
plt.ylabel(r'$\log(\overline{V})$')
plt.savefig('build/a_plot_2.pdf')
plt.clf()

plt.title(r'Frequenzgang eines gekoppelten Verstärkers mit:'+ '\n' 
	+ r'$R_N = 4.7\,\mathrm{k}\Omega;\, R_1 = 100\,\Omega$')
plt.plot(nu_3_log, unp.nominal_values(V_3_log),"xb",label=r'Messpunkte')
plt.plot(x_3,func(x_3,*popt_3),'-k', label=r'Ausgleichsgerade $f_3(x)$')
plt.axhline(y=unp.nominal_values(V_grenz_3_log),color='g',linestyle='--', 
	label=r'$\log(\overline{V}/\sqrt{2})$')
plt.axvline(x=unp.nominal_values(nu_grenz_3_log),color='r',linestyle='--',
	label=r'$\log(\nu_g/\mathrm{Hz})$')
plt.legend(loc='best')
plt.grid()
plt.xlabel(r'$\log(\nu/\mathrm{Hz})$')
plt.ylabel(r'$\log(\overline{V})$')
plt.savefig('build/a_plot_3.pdf')
plt.clf()

plt.title(r'Frequenzgang eines gekoppelten Verstärkers mit:'+ '\n' 
	+ r'$R_N = 1\,\mathrm{M}\Omega;\, R_1 = 4.7\,\mathrm{k}\Omega$')
plt.plot(nu_4_log, unp.nominal_values(V_4_log),"xb",label=r'Messpunkte')
plt.plot(x_4,func(x_4,*popt_4),'-k', label=r'Ausgleichsgerade $f_4(x)$')
plt.axhline(y=unp.nominal_values(V_grenz_4_log),color='g',linestyle='--', 
	label=r'$\log(\overline{V}/\sqrt{2})$')
plt.axvline(x=unp.nominal_values(nu_grenz_4_log),color='r',linestyle='--',
	label=r'$\log(\nu_g/\mathrm{Hz})$')
plt.legend(loc='best')
plt.grid()
plt.xlabel(r'$\log(\nu/\mathrm{Hz})$')
plt.ylabel(r'$\log(\overline{V})$')
plt.savefig('build/a_plot_4.pdf')
plt.clf()