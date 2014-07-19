from numpy import *
from uncertainties import ufloat

a = ufloat(22.860,0.046)
c = 3*10**11

d1 = ufloat(114.2,0.05)
d2 = ufloat(89.6,0.05)
d3 = ufloat(65.0,0.05)
d4 = ufloat(40.8,0.05)

l1 = (d1-d2)*2
l2 = (d2-d3)*2
l3 = (d3-d4)*2

lg = (l1+l2+l3)/3

print("lambda:" + str(lg))

f = c*((1/lg)**2+(1/(2*a))**2)**(1/2)

print("freq:" + str(f))