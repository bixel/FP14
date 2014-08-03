from uncertainties import *
from uncertainties.umath import *
import matplotlib.pyplot as plt
from numpy import pi
from scipy.optimize import curve_fit

d1 = ufloat(96.4,0.005)
d2 = ufloat(93.4,0.005)

m1 = ufloat(115.3,0.005)
m2 = ufloat(90.5,0.005)
m3 = ufloat(66.0,0.005)

l1 = (m1-m2)*2
l2 = (m2-m3)*2

lg = (l1+l2)/2

S1 = str((1+1/sin(pi*(d1-d2)/lg)**2)**(1/2))

S2 = str(lg/(pi*(d1-d2)))

print("lambda: "+str(lg))
print('3.3 genau:' + S1)
print('3.3 genähert' + S2)

A1 = ufloat(20,0.5)
A2 = ufloat(40,0.5)

Ag = A2-A1

S3 = str(10**(Ag/20))

print('3.4:'+str(Ag))
print('3.4:' + S3)
