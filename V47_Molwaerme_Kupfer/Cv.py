#! /usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from codecs import open
from textable import table
from uncertainties import unumpy as unp
from scipy.integrate import quad

#Konstanten
Masse = 342
Molmasse = 63.546
Dichte = 8.92*10**6
Molaresvolumen = 7.11*10**-6
Kompressionsmodul = 140*10**9


# Einlesen der Daten
R0_r, DR0_r, t_r, Dt_r, U_r, DU_r, I_r, DI_r, R1_r, DR1_r = np.genfromtxt('data/daten.txt', unpack=True)
Tt0, Rt, alpha = np.genfromtxt('data/temp.txt', unpack=True)

#Umrechnung in Kelvin und ergänzung der 10^-6 zur Bestimmung der Gleichungen
Tt = Tt0 + 273.15
Talpha = Tt0 + 270
alpha = alpha*10**-6
I_r = I_r * 10**-3
DI_r = DI_r * 10**-3

#Funktionen
def func_alpha(x,a,b,c,d,e,f):
	return a*x**5+b*x**4+c*x**3+d*x**2+e*x+f

def quadratische_funktion(x,a,b,c):
 return a*x**2+b*x+c

def Cp(a,b,c,R0,R1,I,t,U):
	return Molmasse*U*I*t/(Masse*( quadratische_funktion(R1,a,b,c) - quadratische_funktion(R0,a,b,c) ))

def Cp_zu_Cv(T,a,b,c,d,e,f):
	return 9*func_alpha(T,a,b,c,d,e,f)**2*Kompressionsmodul*Molaresvolumen*T

def Temperatur(R0,R1,a,b,c):
	return ( quadratische_funktion(R1,a,b,c) + quadratische_funktion(R0,a,b,c) )/2

#Fehler
R0 = unp.uarray(R0_r, DR0_r)
R1 = unp.uarray(R1_r, DR1_r)
t = unp.uarray(t_r, Dt_r)
U = unp.uarray(U_r, DU_r)
I = unp.uarray(I_r, DI_r)

#Fits
popt_temp1, pcov_temp = curve_fit(quadratische_funktion, Rt, Tt)
popt_temp = unp.uarray(popt_temp1,[pcov_temp[0][0],pcov_temp[1][1],pcov_temp[2][2]])

alpha_plot = alpha / 10**-6
popt_alpha_plot, pcov_alpha_plot = curve_fit(func_alpha, Talpha, alpha_plot)
popt_alpha_plot = unp.uarray(popt_alpha_plot,[pcov_alpha_plot[0][0],pcov_alpha_plot[1][1],pcov_alpha_plot[2][2],pcov_alpha_plot[3][3],pcov_alpha_plot[4][4],pcov_alpha_plot[5][5]])

popt_alpha, pcov_alpha = curve_fit(func_alpha, Talpha, alpha)
popt_alpha = unp.uarray(popt_alpha,[pcov_alpha[0][0],pcov_alpha[1][1],pcov_alpha[2][2],pcov_alpha[3][3],pcov_alpha[4][4],pcov_alpha[5][5]])

#Rechnung b)
T_Cv = Temperatur(R0,R1,popt_temp[0],popt_temp[1],popt_temp[2])
Cp = Cp(popt_temp[0],popt_temp[1],popt_temp[2],R0,R1,I,t,U)
Cv = Cp - Cp_zu_Cv(T_Cv,popt_alpha[0],popt_alpha[1],popt_alpha[2],popt_alpha[3],popt_alpha[4],popt_alpha[5])
alpha_show = func_alpha(T_Cv,popt_alpha_plot[0],popt_alpha_plot[1],popt_alpha_plot[2],popt_alpha_plot[3],popt_alpha_plot[4],popt_alpha_plot[5])

rules = np.linspace(50,350,1000)
result1 = []
result2 = []
Theta_Debye = 345
for i in range(len(rules)):
	result1.append(quad(lambda x: x**4*np.exp(x)/(1-np.exp(x))**2,0,Theta_Debye/rules[i]))
	result2.append(result1[i][0])
	result2[i] *= 9*8.31439 * rules[i]**3 / Theta_Debye**3

#Rechnung c)
Cv_c = unp.uarray([0 for i in range(11)],[0 for i in range(11)])
for i in range(11):
	Cv_c[i] = Cv[i]

Theta_t = unp.uarray([4.1, 3.2, 2.8, 2.4, 2.2, 2.0, 2.0, 2.0, 2.1, 2.4, 2.3],[0.05 for i in range(11)])
Theta_arr = unp.uarray([0 for i in range(11)],[0 for i in range(11)])
summe1 = 0
summe2 = 0
for i in range(len(Theta_t)):
	Theta_arr[i] = Theta_t[i] * T_Cv[i]
	summe1 += Theta_arr[i]
	summe2 += Theta_t[i]
Theta1 = summe1/len(Theta_arr)
Theta2 = summe2/len(Theta_t)

#Rechnung d)
V = Masse/Dichte
vl = 4.7*10**3
vs = 2.26*10**3
h_quer = 1.054571726*10**-34
k_B = 1.3806488*10**-23
Avogadro = 6.02214129*10**23

omega_D = ( 18 * np.pi**2 * Avogadro * Masse/ ( V * Molmasse * (1/vl**3+2/vs**3)))**(1/3)
omega_D_print = omega_D*10**-13
theta_D = h_quer*omega_D/k_B

#Output

(open('build/theta_D_T.tex', 'w', 'utf-8').
 write(r'$\frac{{\Theta_\mathrm{{D}}}}{{T}} = \SI{{{:L}}}{{}}$'.
 format(Theta2)))

(open('build/theta_D.tex', 'w', 'utf-8').
 write(r'$\Theta_\mathrm{{D}} = \SI{{{:L}}}{{\kelvin}}$'.
 format(Theta1)))

(open('build/omega_D_theo.tex', 'w', 'utf-8').
 write(r'$\omega_\mathrm{{D}} = \SI{{{:.3f}e13}}{{1\per\second}}$'.
 format(omega_D_print)))

(open('build/theta_D_theo.tex', 'w', 'utf-8').
 write(r'$\Theta_\mathrm{{D}} = \SI{{{:.3f}}}{{\kelvin}}$'.
 format(theta_D)))

f = open('build/Fit.txt', 'w', 'utf-8')
f.write("Temperatur\n")
for i in range(len(popt_temp)):
	f.write(str(popt_temp[i])+"\n")
f.write('\nAlpha\n')
for i in range(len(popt_alpha)):
	f.write(str(popt_alpha[i])+"\n")
f.close()

f = open('build/Theta.tex', 'w', 'utf-8')
f.write(table([r'$T/\si{\kelvin}$', r'$C_\mathrm{{V}}/\si{\joule\per\mol\per\kelvin}$',
               r'$\frac{\Theta_\mathrm{{D}}}{T}/\si{}$', r'$\Theta_\mathrm{{D}}/\si{\kelvin}$',],
              [T_Cv[:11],Cv[:11],
               Theta_t[:11], Theta_arr[:11]]))
f.close()

f = open('build/Cv.tex', 'w', 'utf-8')
f.write(table([r'$T/\si{\kelvin}$', r'$C_\mathrm{{P}}/\si{\joule\per\mol\per\kelvin}$',
               r'$\alpha \cdot 10^{-6}/\si{1\per\kelvin}$', r'$C_\mathrm{{V}}/\si{\joule\per\mol\per\kelvin}$',],
              [T_Cv,Cp,alpha_show,Cv]))
f.close()

f = open('build/data.tex', 'w', 'utf-8')
f.write(table([r'$R_0/\si{\ohm}$', r'$R_1/\si{\ohm}$',
               r'$t/\si{\second}$', r'$U/\si{\ohm}$',
               r'$I/\si{\milli\ampere}$'],
              [R0,R1,t,U,I]))
f.close()

#Plots
fig = plt.figure()
x = np.linspace(15,115)
plt.plot(Rt,Tt,"x",label="Wertepaare")
plt.plot(x,quadratische_funktion(x,popt_temp1[0],popt_temp1[1],popt_temp1[2]),"-", label="lineare Regression")
plt.xlabel(r"R/"+'$\Omega$')
plt.ylabel(r"T/K")
plt.legend(loc = 2)
plt.savefig("build/Cv_Temperatur.pdf")
plt.close(fig)

fig = plt.figure()
x = np.linspace(65,310)
plt.plot(Talpha, unp.nominal_values(alpha_plot),"+",label="Wertepaare")
plt.plot(x,func_alpha(x,unp.nominal_values(popt_alpha_plot)[0],unp.nominal_values(popt_alpha_plot)[1],unp.nominal_values(popt_alpha_plot)[2],unp.nominal_values(popt_alpha_plot)[3],unp.nominal_values(popt_alpha_plot)[4],unp.nominal_values(popt_alpha_plot)[5]),"-",label="Regressionsfunktion")
plt.xlabel(r"T/K")
plt.ylabel(r"$\alpha \cdot 10^{-6}$ K")
plt.legend(loc=2)
plt.savefig("build/Cv_alpha.pdf")
plt.close(fig)

fig = plt.figure()
x = np.linspace(65,310)
plt.plot(unp.nominal_values(T_Cv), unp.nominal_values(Cv),"+",label="Cv")
plt.plot(rules,result2,"-",label="Theorie")
plt.xlabel(r"T/K")
plt.ylabel(r"$C_V \cdot K \mathrm{mol} / J$  ")
plt.legend(loc=2)
plt.errorbar(unp.nominal_values(T_Cv), unp.nominal_values(Cv), yerr=unp.std_devs(Cv), xerr=unp.std_devs(T_Cv), fmt='b+', label='Messwerte')
plt.savefig("build/Cv_Cv.pdf")
plt.close(fig)