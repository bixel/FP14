from numpy import *
from uncertainties import ufloat

a = ufloat(22.860,0.046)
c = 2.99792458*10**11

d1 = ufloat(114.2,0.005)
d2 = ufloat(89.6,0.005)
d3 = ufloat(65.0,0.005)
d4 = ufloat(40.8,0.005)

l1 = (d1-d2)*2
l2 = (d2-d3)*2
l3 = (d3-d4)*2

lg = (l1+l2+l3)/3

print("lambda:" + str(lg))

f = c*((1/lg)**2+(1/(2*a))**2)**(1/2)

print("freq:" + str(f))

m1 = ufloat(90.5,0.005)
m2 = ufloat(66.0,0.005)
m3 = ufloat(115.3,0.005)
h1 = (m3-m1)*2
h2 = (m1-m2)*2
l = (h1+h2)/2

print("lambda2:" + str(l))