import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit
from codecs import open
from textable import table

C1 = 20e-9
R = 10e3

print('f1 = ' + str(1/(2*np.pi*R*C1)))
print('tau = ' + str(20*R*C1))

t, U = np.genfromtxt('data/scope_19.csv',unpack=True)
N = 38
val = np.zeros(N)
val_x = np.zeros(N)
for i in range(len(t)):
	highest = 0
	if U[i] >= highest:
		highest = U[i]
	for z in range(len(val)):
		if val[z] <= highest:
			val[z] = highest
			val_x[z] = t[i]
			break
val_log = np.log(val)

def func(x,a,b):
	return x/a+b

def func2(x,a,b,c):
	return a*np.exp(x/b)+c

popt, pcov = curve_fit(func,val_x[0:6],val_log[0:6])
popt1, pcov1 = curve_fit(func2,val_x,val)

print(ufloat(popt[0],pcov[0][0]))
print(ufloat(popt[1],pcov[1][1]))

x = np.linspace(val_x[0]-0.0015,val_x[5]+0.00001,1e4)
x1 = np.linspace(val_x[0],val_x[len(val_x)-1],1e4)

plt.plot(x,func(x,*popt),'g-',label=r'Regression $f_h(x)$')
plt.plot(val_x,val_log,'rx',label=r'Messpunkte')

plt.grid()
plt.xlabel(r'$t/\mathrm{s}$')
plt.ylabel(r'$\log{U/\mathrm{V}}$')
plt.legend(loc="best")

plt.savefig('build/h_plot.pdf')

with open('build/h_table.tex', 'w', 'utf-8') as f:
	f.write(table([r'$t/\si{\second}$', r'$U/\si{\volt}$', r'$t/\si{\second}$', r'$U/\si{\volt}$'], 
					[val_x[0:19], val[0:19], val_x[19:38], val[19:38]]))
	print(val_x)