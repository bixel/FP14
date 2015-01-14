import numpy as np
import matplotlib.pyplot as plt

r = 1400
r1 = 10**100
r2 = 1000
r3 = 1400
def g(x,a,b):
	return((1-x/a)*(1-x/b))

x = np.linspace(1,2000,2000)

plt.plot(x,g(x,r,r1),"b")
plt.plot(x,g(x,r,r2),"g")
plt.plot(x,g(x,r,r3),"k")
plt.plot(x,[0 for i in range(2000)],"k")
plt.show()