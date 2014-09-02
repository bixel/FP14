from uncertainties import *
from uncertainties.umath import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def calc_Gu(a,b,c):
	d = ufloat(a,0.05)
	e = ufloat(b,0.005)
	f = ufloat(c,0.05)
	return(d*e/f)

def calc_G(a,b,c):
	return(a*b/c)

gelb_f, gelb_r, gelb_l, gelb_c, gelb_g = np.loadtxt("gelb_rlc.txt", comments="%" ,unpack=True)
lang_f, lang_r, lang_l, lang_c, lang_g = np.loadtxt("lang_rlc.txt", comments="%" , unpack=True)
RG58CU_f, RG58CU_r, RG58CU_l, RG58CU_c, RG58CU_g = np.loadtxt("RG58CU_rlc.txt", comments="%" , unpack=True)

gelb_gu = [0 for i in range(len(gelb_f))]
lang_gu = [0 for i in range(len(gelb_f))]
RG58CU_gu = [0 for i in range(len(gelb_f))]

for i in range(len(gelb_f)):
	gelb_gu[i] = str(calc_Gu(gelb_r[i],gelb_c[i],gelb_l[i]))
	lang_gu[i] = str(calc_Gu(lang_r[i],lang_c[i],lang_l[i]))
	RG58CU_gu[i] = str(calc_Gu(RG58CU_r[i],RG58CU_c[i],RG58CU_l[i]))

gelb_g = np.zeros(len(gelb_f))
lang_g = np.zeros(len(gelb_f))
RG58CU_g = np.zeros(len(gelb_f))

for i in range(len(gelb_f)):
	gelb_g[i] = calc_G(gelb_r[i],gelb_c[i],gelb_l[i])
	lang_g[i] = calc_G(lang_r[i],lang_c[i],lang_l[i])
	RG58CU_g[i] = calc_G(RG58CU_r[i],RG58CU_c[i],RG58CU_l[i])


fig = plt.figure()
plt.plot(gelb_f, gelb_r, 'r-', label="R")
plt.plot(gelb_f, lang_r, 'b-', label="R")
plt.plot(gelb_f, RG58CU_r, 'g-', label="R")
plt.legend(loc=1)
plt.ylim([5,45])
plt.savefig("R.pdf")
plt.close(fig)

fig = plt.figure()
plt.plot(lang_f, gelb_l, 'r-', label="L")
plt.plot(lang_f, lang_l, 'b-', label="L")
plt.plot(lang_f, RG58CU_l, 'g-', label="L")
plt.legend(loc=2)
plt.ylim([0,40])
plt.savefig("L.pdf")
plt.close(fig)

fig = plt.figure()
plt.plot(RG58CU_f, gelb_c, 'r-', label="C")
plt.plot(RG58CU_f, lang_c, 'b-', label="C")
plt.plot(RG58CU_f, RG58CU_c, 'g-', label="C")
plt.legend(loc=1)
plt.ylim([0,12])
plt.savefig("C.pdf")
plt.close(fig)

fig = plt.figure()
plt.plot(RG58CU_f, gelb_g, 'r-', label="G")
plt.plot(RG58CU_f, lang_g, 'b-', label="G")
plt.plot(RG58CU_f, RG58CU_g, 'g-', label="G")
plt.legend(loc=2)
plt.savefig("G.pdf")
plt.close(fig)
