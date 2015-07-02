import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit
from codecs import open
from textable import table

c = 47e-9
r = 10e3

def func(x,a,b):
	return	a*x+b

#d)
nu, U_raw, dU = np.genfromtxt('data/data_d.txt',unpack=True)

U = unp.uarray(U_raw,dU)

U_log = np.log(U_raw)
nu_log = np.log(nu)

#Fit
popt, pcov = curve_fit(func, nu_log, U_log)

#plot
x = np.linspace(nu_log[0]-0.3,nu_log[len(nu_log)-1]+0.3,1e5)
plt.plot(x,func(x,*popt),'k-',label=r'Regression')

plt.plot(nu_log,U_log,'bx',label=r'Messpunkte')

plt.legend(loc='best')
plt.title(r'Frequenzgangs eines Umkehrintegrators')
plt.xlabel(r'$\log(\nu/\mathrm{Hz})$')
plt.ylabel(r'$\log(U/\mathrm{V})$')
plt.grid()

plt.savefig('build/d_plot.pdf')
plt.clf()

#e)
nu, U_raw, dU = np.genfromtxt('data/data_e.txt',unpack=True)

U = unp.uarray(U_raw,dU)

U_log = np.log(U_raw)
nu_log = np.log(nu)

#fit
popt, pcov = curve_fit(func, nu_log[3:10], U_log[3:10])

#plot
x = np.linspace(nu_log[3]-0.3,nu_log[9]+0.3,1e5)
plt.plot(x,func(x,*popt),'k-',label=r'Regression')

plt.plot(nu_log,U_log,'bx',label=r'Messpunkte')

plt.legend(loc='best')
plt.title(r'Frequenzgangs eines Umkehrdifferentiators')
plt.xlabel(r'$\log(\nu/\mathrm{Hz})$')
plt.ylabel(r'$\log(U/\mathrm{V})$')
plt.grid()

plt.savefig('build/e_plot.pdf')
plt.clf()
