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
	return a*x+b

def func2(x,a,b,c):
	return a*np.exp(x/b)+c

popt, pcov = curve_fit(func,val_x[5:32],val_log[5:32])
popt1, pcov1 = curve_fit(func2,val_x,val)

print(1/popt[0])


x = np.linspace(val_x[5],val_x[30],1e4)
x1 = np.linspace(val_x[0],val_x[len(val_x)-1],1e4)

# plt.plot(x1,func2(x1,*popt1),'y-')
# plt.plot(val_x,val,'rx')

plt.plot(x,func(x,*popt),'g-')
plt.plot(val_x[5:32],val_log[5:32],'rx')

plt.savefig('lol.pdf')