import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit

# Import Data
nu_1, V_1, dV_1 = np.genfromtxt("data/data_a_1.txt",unpack=True)
nu_2, V_2, dV_2 = np.genfromtxt("data/data_a_2.txt",unpack=True)
nu_3, V_3, dV_3 = np.genfromtxt("data/data_a_3.txt",unpack=True)
nu_4, V_4, dV_4 = np.genfromtxt("data/data_a_4.txt",unpack=True)

nu_1_log, V_1_log, dV_1_log = np.log(nu_1), np.log(V_1), np.log(dV_1)
nu_2_log, V_2_log, dV_2_log = np.log(nu_2), np.log(V_2), np.log(dV_2)
nu_3_log, V_3_log, dV_3_log = np.log(nu_3), np.log(V_3), np.log(dV_3)
nu_4_log, V_4_log, dV_4_log = np.log(nu_4), np.log(V_4), np.log(dV_4)

# Fit Data
def func(x,a,b):
	return a*x+b

popt_1, pcov_1 = curve_fit(func,nu_1_log,V_1_log)
popt_2, pcov_2 = curve_fit(func,nu_2_log[6:14],V_2_log[6:14])
popt_3, pcov_3 = curve_fit(func,nu_3_log[12:18],V_3_log[12:18])
popt_4, pcov_4 = curve_fit(func,nu_4_log[11:18],V_4_log[11:18])

# Calc Grenzfrequenz

V_grenz_1 = ufloat(V_1[0],dV_1[0]) / np.sqrt(2)
V_grenz_2 = ufloat(V_2[0],dV_2[0]) / np.sqrt(2)
V_grenz_3 = ufloat(V_3[0],dV_3[0]) / np.sqrt(2)
V_grenz_4 = ufloat(V_4[0],dV_4[0]) / np.sqrt(2)

V_grenz_1_log = unp.log(V_grenz_1)
V_grenz_2_log = unp.log(V_grenz_2)
V_grenz_3_log = unp.log(V_grenz_3)
V_grenz_4_log = unp.log(V_grenz_4)

nu_grenz_1_log = ( V_grenz_1_log - popt_1[1] ) / popt_1[0]
nu_grenz_2_log = ( V_grenz_2_log - popt_2[1] ) / popt_2[0]
nu_grenz_3_log = ( V_grenz_3_log - popt_3[1] ) / popt_3[0]
nu_grenz_4_log = ( V_grenz_4_log - popt_4[1] ) / popt_4[0]

nu_grenz_1 = unp.exp(nu_grenz_1_log)
nu_grenz_2 = unp.exp(nu_grenz_2_log)
nu_grenz_3 = unp.exp(nu_grenz_3_log)
nu_grenz_4 = unp.exp(nu_grenz_4_log)

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
plt.plot(nu_1_log, V_1_log,"xb",label=r'Messpunkte')
plt.plot(x_1,func(x_1,*popt_1),'-k', label=r'Ausgleichsgerade $f_1(x)$')
plt.legend(loc='best')
plt.grid()
plt.xlabel(r'$\log(\nu/\mathrm{Hz})$')
plt.ylabel(r'$\log(\overline{V})$')
plt.savefig('build/plot_a_1.pdf')
plt.clf()

plt.title(r'Frequenzgang eines gekoppelten Verstärkers mit:'+ '\n' 
	+ r'$R_N = 120\,\mathrm{k}\Omega;\, R_1 = 100\,\Omega$')
plt.plot(nu_2_log, V_2_log,"xb",label=r'Messpunkte')
plt.plot(x_2,func(x_2,*popt_2),'-k', label=r'Ausgleichsgerade $f_2(x)$')
plt.axhline(y=unp.nominal_values(V_grenz_2_log),color='g',linestyle='--', 
	label=r'$\log(\overline{V}/\sqrt{2})$')
plt.axvline(x=unp.nominal_values(nu_grenz_2_log),color='r',linestyle='--',
	label=r'$\log(\nu_g/\mathrm{Hz})$')
plt.legend(loc='best')
plt.grid()
plt.xlabel(r'$\log(\nu/\mathrm{Hz})$')
plt.ylabel(r'$\log(\overline{V})$')
plt.savefig('build/plot_a_2.pdf')
plt.clf()

plt.title(r'Frequenzgang eines gekoppelten Verstärkers mit:'+ '\n' 
	+ r'$R_N = 4.7\,\mathrm{k}\Omega;\, R_1 = 100\,\Omega$')
plt.plot(nu_3_log, V_3_log,"xb",label=r'Messpunkte')
plt.plot(x_3,func(x_3,*popt_3),'-k', label=r'Ausgleichsgerade $f_3(x)$')
plt.axhline(y=unp.nominal_values(V_grenz_3_log),color='g',linestyle='--', 
	label=r'$\log(\overline{V}/\sqrt{2})$')
plt.axvline(x=unp.nominal_values(nu_grenz_3_log),color='r',linestyle='--',
	label=r'$\log(\nu_g/\mathrm{Hz})$')
plt.legend(loc='best')
plt.grid()
plt.xlabel(r'$\log(\nu/\mathrm{Hz})$')
plt.ylabel(r'$\log(\overline{V})$')
plt.savefig('build/plot_a_3.pdf')
plt.clf()

plt.title(r'Frequenzgang eines gekoppelten Verstärkers mit:'+ '\n' 
	+ r'$R_N = 1\,\mathrm{M}\Omega;\, R_1 = 4.7\,\mathrm{k}\Omega$')
plt.plot(nu_4_log, V_4_log,"xb",label=r'Messpunkte')
plt.plot(x_4,func(x_4,*popt_4),'-k', label=r'Ausgleichsgerade $f_4(x)$')
plt.axhline(y=unp.nominal_values(V_grenz_4_log),color='g',linestyle='--', 
	label=r'$\log(\overline{V}/\sqrt{2})$')
plt.axvline(x=unp.nominal_values(nu_grenz_4_log),color='r',linestyle='--',
	label=r'$\log(\nu_g/\mathrm{Hz})$')
plt.legend(loc='best')
plt.grid()
plt.xlabel(r'$\log(\nu/\mathrm{Hz})$')
plt.ylabel(r'$\log(\overline{V})$')
plt.ylim(-6,2)
plt.savefig('build/plot_a_4.pdf')
plt.clf()