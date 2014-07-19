import matplotlib.pyplot as plt
from numpy import *
from scipy.optimize import curve_fit

db = array([0,2,4,6,8,10])
mm = array([1.00,1.40,1.89,2.11,2.34,2.48])

mm2 = array([0,1,2,3,4,5])
db2 = array([0,2,8,16,29,46])

def func(x,a,b,c):
	return a*x**2+b*x+c

params1 = curve_fit(func,mm2,db2)
params2 = curve_fit(func,mm,db)

[a,b,c] = params1[0]
[d,e,f] = params2[0]

print(params1[0])
print(params2[0])

x=arange(0,5,0.01)
plt.plot(x,a*x**2+b*x+c, 'r-',label="Eichkurve")
plt.plot(mm,db,'gx',label="messpunkte")
plt.plot(x,d*x**2+e*x+f, 'g-', label="Regression")
plt.ylim([0,50])
plt.legend(loc=2)
plt.xlabel("Mikrometereinstellung [mm]")
plt.ylabel('Dämpfung [dB]')
plt.savefig("..\pic\Daempfung.pdf")