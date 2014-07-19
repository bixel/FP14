from numpy import *
from uncertainties import ufloat

f1 = ufloat(9024.0,0.5)
f2 = ufloat(8982.5,0.5)
v1 = ufloat(225,5)
v2 = ufloat(205,5)

y = (f1-f2)/(v1-v2)

print(y)