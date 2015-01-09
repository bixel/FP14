#! /usr/bin/env python3
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp
#Konstanten
Masse = 342
Molmasse = 63.546
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

popt_alpha, pcov_alpha = curve_fit(func_alpha, Talpha, alpha)


#Rechnung
T_Cv = Temperatur(R0,R1,popt_temp1[0],popt_temp1[1],popt_temp1[2])

Cv = Cp(popt_temp[0],popt_temp[1],popt_temp[2],R0,R1,I,t,U) - Cp_zu_Cv(T_Cv,popt_alpha[0],popt_alpha[1],popt_alpha[2],popt_alpha[3],popt_alpha[4],popt_alpha[5])
print(Cv)

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
plt.plot(Talpha, alpha_plot,"+",label="Wertepaare")
plt.plot(x,func_alpha(x,popt_alpha_plot[0],popt_alpha_plot[1],popt_alpha_plot[2],popt_alpha_plot[3],popt_alpha_plot[4],popt_alpha_plot[5]),"-",label="Regressionsfunktion")
plt.xlabel(r"T/K")
plt.ylabel(r"$\alpha \cdot 10^{-6}$ K")
plt.savefig("build/Cv_alpha.pdf")
plt.close(fig)

fig = plt.figure()
x = np.linspace(65,310)
plt.plot(unp.nominal_values(T_Cv), unp.nominal_values(Cv),"+",label="Cv")
plt.xlabel(r"T/K")
plt.ylabel(r"$C_V \cdot K \mathrm{mol} / J$  ")
plt.errorbar(unp.nominal_values(T_Cv), unp.nominal_values(Cv), yerr=unp.std_devs(Cv), xerr=unp.std_devs(T_Cv), fmt='b+', label='Messwerte')
plt.savefig("build/Cv_Cv.pdf")
plt.close(fig)