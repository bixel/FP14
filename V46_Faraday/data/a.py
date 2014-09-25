from uncertainties import *
from uncertainties.umath import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats

d, B = np.loadtxt("a.txt", comments="%" ,unpack=True)

def func(x,a,b,c):
	return(a*np.exp(-1/2*((x-b)/c)**2))

def func2(x,a,b,c):
	return(a*b**2/((x-c)**2+b**2))

W = curve_fit(func, d, B)
a1, b1, c1 = W[0]
print(a1)
print(b1)
print(c1)

K = curve_fit(func2, d, B)
a2, b2, c2 = K[0]
print(a2)
print(b2)
print(c2)

go = np.linspace(0,60,1000)

plt.plot(go,func(go,a1,b1,c1),"b-", label="Gauß-Funktion")
plt.plot(go,func2(go,a2,b2,c2),"r-", label="Lorentz-Funktion")
plt.axvline(x=b1, ymin=0, ymax=30, color="y", label="HP Gauß")
plt.axvline(x=c2, ymin=0, ymax=30, color="k", label="HP Lorentz")
plt.plot(d,B,".g", label="Meßpunkte")
plt.ylabel("B(z)[mT]")
plt.xlabel("z[mm]")
plt.legend()
#plt.show()
plt.savefig("gauß.pdf")