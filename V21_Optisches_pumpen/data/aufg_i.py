import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp
from textable import table

def func_exp(x, a, b, c, d):
	return a * np.exp(-b*(x + c)) + d

def func_exp2(x, a, b, c):
	return a * np.exp(-b*x) + c

def func_hyp(x, a, b, c):
	return a + b / (x - c)

DELTA_T = 40 * 10**-6
DELTA_EX = 5 * 10**-7
DELTA_EY = 0.05

expX1, expY1 = np.genfromtxt('ALL0002/F0002CH2.CSV', unpack=True, delimiter=',', usecols=(3,4))
expX2, expY2 = np.genfromtxt('ALL0003/F0003CH2.CSV', unpack=True, delimiter=',', usecols=(3,4))

expX1E = unp.uarray(expX1, DELTA_EX)
expX2E = unp.uarray(expX2, DELTA_EX)
expY1E = unp.uarray(expY1, DELTA_EY)
expY2E = unp.uarray(expY2, DELTA_EY)

V1, tOne1, tTwo1, number1 = np.genfromtxt('aufg_i_peak1.dat', unpack=True)
V2, tOne2, tTwo2, number2 = np.genfromtxt('aufg_i_peak2.dat', unpack=True)

tOne1 = tOne1 * 10**-3
tTwo1 = tTwo1 * 10**-3
tOne2 = tOne2 * 10**-3
tTwo2 = tTwo2 * 10**-3

tOne1E = unp.uarray(tOne1, DELTA_T)
tTwo1E = unp.uarray(tTwo1, DELTA_T)
tOne2E = unp.uarray(tOne2, DELTA_T)
tTwo2E = unp.uarray(tTwo2, DELTA_T)

period1 = (tTwo1 - tOne1) / number1
period2 = (tTwo2 - tOne2) / number2
period1E = (tTwo1E - tOne1E) / number1
period2E = (tTwo2E - tOne2E) / number2

with open('i_raw.tex', 'w') as f:
	f.write(
		table([
				r'$V_\text{87} / \si{\volt}$',
				r'$t_\text{1,87} / \si{\micro\second}$',
				r'$t_\text{2,87} / \si{\micro\second}$',
				r'$T_\text{87} / \si{\micro\second}$',
				r'#Perioden$_\text{87}$',
				r'$V_\text{85} / \si{\volt}$',
				r'$t_\text{1,85} / \si{\micro\second}$',
				r'$t_\text{2,85} / \si{\micro\second}$',
				r'#Perioden$_\text{85}$',
				r'$T_\text{85} / \si{\micro\second}$'
				],[
					V1,
					tOne1E*10**6,
					tTwo1E*10**6,
					number1,
					period1E*10**6,
					V2,
					tOne2E*10**6,
					tTwo2E*10**6,
					number2,
					period2E*10**6
				]
				)
			)

poptE1, pcovE1 = curve_fit(
	func_exp2, expX1[600:-1000], expY1[600:-1000], 
	p0=(-1000000, 110, 9.3)
	)
print(poptE1)
print(expY1[700:-1])
print(np.sqrt(np.diag(pcovE1)))
# with open('i_fit_em_87.tex', 'w') as f:
# 	f.write(r'\SI{%1.4e}{\tesla\per\hertz}'%(popt1[0]))
# with open('i_fit_eb_87.tex', 'w') as f:
# 	f.write(r'\SI{%1.4e}{\tesla\per\hertz}'%(popt1[1]))
x=np.linspace(expX1[600], expX1[-1], 10**5)
plt.plot(x, func_exp2(x, *poptE1))
plt.plot(
	expX1[600:-1], expY1[600:-1], 'kx', markersize=2
	)
plt.xlim((expX1[600]*99/100,expX1[-1]*101/100))
plt.grid()
plt.savefig('fit_E_87.pdf')
plt.clf()

poptE2, pcovE2 = curve_fit(
	func_exp, expX2[800:-1], expY2[800:-1],
	p0=(-1.4, 1.3, -0.09, 1.3)
	)
x=np.linspace(expX2[800], expX2[-1], 10**5)
plt.plot(x, func_exp(x, *poptE2))
plt.plot(
	expX2[800:-1], expY2[800:-1], 'kx', markersize=2
	)
plt.xlim((expX2[800]*99/100,expX2[-1]*101/100))
plt.grid()
plt.savefig('fit_E_85.pdf')
plt.clf()

poptT1, pcovT1 = curve_fit(
	func_hyp, V1, period1, p0=(0.00015, 0.0015, 0)
	)
x=np.linspace(V1[0], V1[-1], 10**5)
plt.plot(x, func_hyp(x, *poptT1))
plt.errorbar(
	V1, period1, yerr=unp.std_devs(period1E), fmt='k.', markersize=1
	)
plt.plot(V1, period1, 'rx')
plt.xlim((V1[0]*90/100, V1[-1]*101/100))
plt.grid()
plt.savefig('fit_T_87.pdf')
plt.clf()

poptT2, pcovT2 = curve_fit(
	func_hyp, V2, period2, p0=(0.00015, 0.0015, 0)
	)
x=np.linspace(V2[0], V2[-1], 10**5)
plt.plot(x, func_hyp(x, *poptT2))
plt.errorbar(
	V2, period2, yerr=unp.std_devs(period2E), fmt='k.', markersize=1
	)
plt.plot(V1, period2, 'rx')
plt.xlim((V2[0]*90/100, V2[-1]*101/100))
plt.grid()
plt.savefig('fit_T_85.pdf')
plt.clf()
