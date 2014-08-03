import matplotlib.pyplot as plt
from numpy import *
from scipy.optimize import curve_fit

db = array([0,2,4,6,8,10])
mm = array([1.00-1,1.40-1,1.89-1,2.11-1,2.34-1,2.48-1])

mm2 = array([0,1,2,3,4,5])
db2 = array([0,2,8,16,29,46])

db3 = array([0,2,4,6,8,10])
mm3 = array([1.00,1.40,1.89,2.11,2.34,2.48])

def func(x,a,b,c,d,e):
	return a*(x-d)**2+b*(x-e)+c

params1 = curve_fit(func,mm2,db2)
params2 = curve_fit(func,mm,db)
params3 = curve_fit(func,mm3,db3)

[a,b,c,g,m] = params1[0]
[d,e,f,h,n] = params2[0]
[i,j,k,l,o] = params3[0]
print(params1[0])
print(params2[0])
print(params3[0])

x=arange(0,5,0.01)
plt.plot(x,func(x,a,b,c,g,m), 'r-',label="Eichkurve")

plt.plot(mm3,db3,'gx',label="Messpunkte")
plt.plot(x,func(x,i,j,k,l,o), 'g-', label="Regression mit Offset")

plt.plot(mm,db,'bx',label="Messpunkte ohne Offset")
plt.plot(x,func(x,d,e,f,h,n), 'b-', label="Regression ohne Offset")

plt.ylim([0,50])
plt.legend(loc=2)
plt.xlabel("Mikrometereinstellung [mm]")
plt.ylabel('Dämpfung [dB]')
plt.savefig("..\pic\Daempfung.pdf")