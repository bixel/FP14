import matplotlib.pyplot as plt
from numpy import *
from scipy.optimize import curve_fit

A1 = array([200,215,230])
A2 = array([120,135,150])
A3 = array([70,80,90])

F1 = array([0,600,0])
F2 = array([0,600,0])
F3 = array([0,450,0])

def func1(x,a):
	return a*exp(-(x-215)**2/2)

def func2(x,a):
	return a*exp(-(x-135)**2/2)

def func3(x,a):
	return a*exp(-(x-80)**2/3)

params1 = curve_fit(func1, A1, F1)
params2 = curve_fit(func2, A2, F2)
params3 = curve_fit(func3, A3, F3)

[a]=params1[0]
[c]=params2[0]
[e]=params3[0]

print(params1)
print(params2)
print(params3)

x=arange(70,230,0.1)
plt.plot(x,a*exp(-(x-215)**2/30), 'g-')
plt.plot(x,c*exp(-(x-135)**2/25), 'r-')
plt.plot(x,e*exp(-(x-80)**2/11), 'b-')
plt.plot(A1,F1,'g.')
plt.plot(A2,F2,'r.')
plt.plot(A3,F3,'b.')
plt.ylim([0,700])
plt.xlabel("Spannung [V]")
plt.ylabel("Spannung [mV]")
plt.savefig("..\pic\ModenDiagramm.pdf")
