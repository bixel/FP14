import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit
from codecs import open
from textable import table

k_b = 1.3806488e-23
e_0 = 1.602176565e-19

const = e_0 / k_b

U_in_log, U_out_log , dU_out_log= np.genfromtxt('data/data_i_log.txt',unpack=True)
U_in_exp, U_out_exp, dU_out_exp = np.genfromtxt('data/data_i_exp.txt',unpack=True)

U_out_exp_rescale = -U_out_exp

def func(x,a,b):
	return a*x + b

def func_exp(x,a,b,c):
	return a*np.exp(b*x) + c

popt_log, pcov_log = curve_fit(func, np.log(U_in_log[1:]), U_out_log[1:])
popt_exp, pcov_exp = curve_fit(func, U_in_exp[1:], U_out_exp_rescale[1:])

print(str(ufloat(popt_log[0],pcov_log[0][0])) + ' ' + str(ufloat(popt_log[1],pcov_log[1][1])))
print(str(ufloat(popt_exp[0],pcov_exp[0][0])) + ' ' + str(ufloat(popt_exp[1],pcov_exp[1][1])))

#Berechnung von T
T_log = popt_log[0] * const
T_exp = const / popt_exp[0]

print(T_log)
print(T_exp)

x_log = np.linspace(np.log(U_in_log[1])-0.4, np.log(U_in_log[len(U_in_log)-1])+0.4, 1e3)
x_exp = np.linspace(U_in_exp[1]-0.4, U_in_exp[15]+0.4, 1e3)

plt.plot(np.log(U_in_log[1:]), U_out_log[1:],'xb',label=r'Messpunkte')
plt.plot(x_log, func(x_log, *popt_log), 'k-', label=r'Ausgleichsgerade $f_i(x)$')

plt.grid()
plt.title(r'Spannungscharakteristik des verwendeten Logarithmierers' + '\n')
plt.xlabel(r'$\log(U_\mathrm{ein}/\mathrm{V})$')
plt.ylabel(r'$U_\mathrm{aus}/\mathrm{V}$')
plt.legend(loc='best')

plt.savefig('build/i_log_plot.pdf')
plt.clf()

plt.plot(U_in_exp[1:], U_out_exp_rescale[1:],'xb', label='Messpunkte')
# plt.plot(U_in_exp[5:], np.log(U_out_exp_rescale[5:]),'xb', label='Messpunkte')
# plt.plot(x_exp, func(x_exp, *popt_exp), 'k-', label='Regression')

plt.grid()
plt.title(r'Spannungscharakteristik des verwendeten Exponentierers' + '\n')
plt.ylabel(r'$U_\mathrm{aus}/\mathrm{V}$')
plt.xlabel(r'$U_\mathrm{ein}/\mathrm{V}$')
plt.legend(loc='best')

plt.savefig('build/i_exp_plot.pdf')
plt.clf()