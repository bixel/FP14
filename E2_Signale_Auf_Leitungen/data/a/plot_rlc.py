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

l1 = 10
l2 = 20
l3 = 100

gelb_f, gelb_r, gelb_l, gelb_c, gelb_g = np.loadtxt("gelb_rlc.txt", comments="%" ,unpack=True)
lang_f, lang_r, lang_l, lang_c, lang_g = np.loadtxt("lang_rlc.txt", comments="%" , unpack=True)
RG58CU_f, RG58CU_r, RG58CU_l, RG58CU_c, RG58CU_g = np.loadtxt("RG58CU_rlc.txt", comments="%" , unpack=True)

gelb_gu = [ufloat(0,0) for i in range(len(gelb_f))]
lang_gu = [ufloat(0,0)  for i in range(len(gelb_f))]
RG58CU_gu = [ufloat(0,0)  for i in range(len(gelb_f))]

for i in range(len(gelb_f)):
	gelb_gu[i] = calc_Gu(gelb_r[i],gelb_c[i],gelb_l[i])/l1
	lang_gu[i] = calc_Gu(lang_r[i],lang_c[i],lang_l[i])/l3
	RG58CU_gu[i] = calc_Gu(RG58CU_r[i],RG58CU_c[i],RG58CU_l[i])/l2

gelb_g = np.zeros(len(gelb_f))
lang_g = np.zeros(len(gelb_f))
RG58CU_g = np.zeros(len(gelb_f))

for i in range(len(gelb_f)):
	gelb_g[i] = calc_G(gelb_r[i],gelb_c[i],gelb_l[i])/l1
	lang_g[i] = calc_G(lang_r[i],lang_c[i],lang_l[i])/l3
	RG58CU_g[i] = calc_G(RG58CU_r[i],RG58CU_c[i],RG58CU_l[i])/l2

for i in range(len(gelb_f)):
	gelb_r[i] = gelb_r[i]/l1
	gelb_l[i] = gelb_l[i]/l1
	gelb_c[i] = gelb_c[i]/l1
	lang_r[i] = lang_r[i]/l3
	lang_l[i] = lang_l[i]/l3
	lang_c[i] = lang_c[i]/l3
	RG58CU_r[i] = RG58CU_r[i]/l2
	RG58CU_l[i] = RG58CU_l[i]/l2
	RG58CU_c[i] = RG58CU_c[i]/l2

fig = plt.figure()
plt.plot(gelb_f, gelb_r, 'r-', label="R RG58CU gelb")
plt.plot(gelb_f, lang_r, 'b-', label="R M17/028 RG058")
plt.plot(gelb_f, RG58CU_r, 'g-', label="R RG58CU")
plt.legend(loc=1)
plt.ylim([0,2.5])
plt.xlabel("f[kHz]")
plt.ylabel("R[Ohm/m]")
plt.savefig("R.pdf")
plt.close(fig)

fig = plt.figure()
plt.plot(lang_f, gelb_l, 'r-', label="L RG58CU gelb")
plt.plot(lang_f, lang_l, 'b-', label="L M17/028 RG058")
plt.plot(lang_f, RG58CU_l, 'g-', label="L RG58CU")
plt.legend(loc=1)
plt.ylim([0,1])
plt.xlabel("f[kHz]")
plt.ylabel("L[µH/m]")
plt.savefig("L.pdf")
plt.close(fig)

fig = plt.figure()
plt.plot(RG58CU_f, gelb_c, 'r-', label="C RG58CU gelb")
plt.plot(RG58CU_f, lang_c, 'b-', label="C M17/028 RG058")
plt.plot(RG58CU_f, RG58CU_c, 'g-', label="C RG58CU")
plt.legend(loc=1)
plt.ylim([0,0.3])
plt.xlabel("f[kHz]")
plt.ylabel("C[nF/m]")
plt.savefig("C.pdf")
plt.close(fig)

fig = plt.figure()
plt.plot(RG58CU_f, gelb_g, 'r-', label="G RG58CU gelb")
plt.plot(RG58CU_f, lang_g, 'b-', label="G M17/028 RG058")
plt.plot(RG58CU_f, RG58CU_g, 'g-', label="G RG58CU")
plt.legend(loc=1)
plt.xlabel("f[kHz]")
plt.ylabel("G[mS/m]")
plt.savefig("G.pdf")
plt.close(fig)
