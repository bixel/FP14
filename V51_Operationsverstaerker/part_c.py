import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit
from codecs import open
from textable import table

U_g = ufloat(1,0.1)
R_v = 10e3
R_N = 100e3

I = U_g / R_v
print(I)
print(I / 1e-6)
with open('build/c_I.tex', 'w') as f:
	f.write(r'\SI{{{:L}}}{{\micro\ampere}}'.format(I / 1e-6))

nu, U_e_raw, dU_e, U_a_raw, dU_a = np.genfromtxt('data/data_c.txt', unpack=True)

U_e = unp.uarray(U_e_raw,dU_e)
U_a = unp.uarray(U_a_raw,dU_a)

U_a_m = I * R_N

print(U_a_m)
with open('build/c_U_a.tex', 'w') as f:
	f.write(r'\num{{{:L}}}'.format( U_a_m ))

R_e = np.array(U_e / I)
V = np.array(R_N / R_e)

with open('build/c_table.tex', 'w', 'utf-8') as f:
	f.write(table([r'$\nu/\si{\hertz}$', r'$U_e/\si{\volt}$', r'$U_a/\si{\volt}$', 
					r'$R_e/\si{\ohm}$', r'$V$'], 
					[nu, U_e, U_a, R_e, V]))

def func(x,a,b):
	return a*x+b

popt_1, pcov_1 = curve_fit(func,np.log(nu[:5]),np.log(unp.nominal_values(R_e[:5])))
popt_2, pcov_2 = curve_fit(func,np.log(nu[:5]),np.log(unp.nominal_values(V[:5])))
popt_3, pcov_3 = curve_fit(func,np.log(nu[:9]),np.log(unp.nominal_values(R_e[:9])))
popt_4, pcov_4 = curve_fit(func,np.log(nu[:9]),np.log(unp.nominal_values(V[:9])))

print(str(ufloat(popt_1[0],pcov_1[0][0])) + '	' + str(ufloat(popt_1[1],pcov_1[1][1]))) 
print(str(ufloat(popt_2[0],pcov_2[0][0])) + '	' + str(ufloat(popt_2[1],pcov_2[1][1]))) 
print(str(ufloat(popt_3[0],pcov_3[0][0])) + '	' + str(ufloat(popt_3[1],pcov_3[1][1]))) 
print(str(ufloat(popt_4[0],pcov_4[0][0])) + '	' + str(ufloat(popt_4[1],pcov_4[1][1]))) 

x1 = np.linspace(np.log(nu[0]),np.log(nu[4]),1e5)
x2 = np.linspace(np.log(nu[0]),np.log(nu[8]),1e5)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.log(nu[:5]), np.log(unp.nominal_values(R_e[:5])),"xb",label=r'Messpunkte $R_e$')
ax.plot(x1, func(x1,*popt_1),'k-', label=r'Regression $f_{c,R,1}$')
ax2 = ax.twinx()
ax2.plot(np.log(nu[:5]), np.log(unp.nominal_values(V[:5])),"xr",label=r'Messpunkte $V$')
ax2.plot(x1, func(x1,*popt_2),'y-', label=r'Regression $f_{c,V,1}$')
ax.legend(loc='center left')
ax2.legend(loc='center right')
ax.grid()
ax.set_title(r'Eingangswiderstand und Leerlaufverstärkung des Operationsverstärkers' + '\n')
ax.set_xlabel(r'$\log(\nu/\mathrm{Hz})$')
ax.set_ylabel(r'$\log(R_e/\mathrm{\Omega})$')
ax2.set_ylabel(r'$\log(V)$')
plt.savefig('build/c_plot.pdf')
plt.close(fig)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(np.log(nu), np.log(unp.nominal_values(R_e)),"xb",label=r'Messpunkte $R_e$')
ax.plot(x2, func(x2,*popt_3),'k-',label=r'Regression $f_{c,R,2}$')
ax2 = ax.twinx()
ax2.plot(np.log(nu), np.log(unp.nominal_values(V)),"xr",label=r'Messpunkte $V$')
ax2.plot(x2, func(x2,*popt_4),'y-', label=r'Regression $f_{c,V,2}$')
ax2.axvline(x=np.log(20e3),color='g',linestyle='--', label=r'Grenzfrequenz $\nu_g$')
ax.legend(loc='center left')
ax2.legend(loc='center right')
ax.grid()
ax.set_title(r'Eingangswiderstand und Leerlaufverstärkung des Operationsverstärkers' + '\n')
ax.set_xlabel(r'$\log(\nu/\mathrm{Hz})$')
ax.set_ylabel(r'$\log(R_e/\mathrm{\Omega})$')
ax2.set_ylabel(r'$\log(V)$')
plt.savefig('build/c_plot_full.pdf')
plt.close(fig)