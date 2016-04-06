import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp
from textable import table


def func_exp(x, a, b, c, d):
    return a * np.exp(b*x + c) + d


def func_linear(x, a, b):
    return a*x + b


def func_hyp(x, a, b, c):
    return a + b / (x - c)

DELTA_T = 40 * 10**-6
DELTA_EX = 5 * 10**-7
DELTA_EY = 0.05

expX1, expY1 = np.genfromtxt(
    'data/ALL0002/F0002CH2.CSV',
    unpack=True,
    delimiter=',',
    usecols=(3, 4)
    )
expX2, expY2 = np.genfromtxt(
    'data/ALL0003/F0003CH2.CSV',
    unpack=True,
    delimiter=',',
    usecols=(3, 4)
    )

expX1E = unp.uarray(expX1, DELTA_EX)
expX2E = unp.uarray(expX2, DELTA_EX)
expY1E = unp.uarray(expY1, DELTA_EY)
expY2E = unp.uarray(expY2, DELTA_EY)

V1, tOne1, tTwo1, number1 = np.genfromtxt('data/aufg_i_peak2.dat', unpack=True)
V2, tOne2, tTwo2, number2 = np.genfromtxt('data/aufg_i_peak1.dat', unpack=True)

tOne1 = tOne1 * 10**-3
tTwo1 = tTwo1 * 10**-3
tOne2 = tOne2 * 10**-3
tTwo2 = tTwo2 * 10**-3

tOne1E = unp.uarray(tOne1, DELTA_T)
tTwo1E = unp.uarray(tTwo1, DELTA_T)
tOne2E = unp.uarray(tOne2, DELTA_T)
tTwo2E = unp.uarray(tTwo2, DELTA_T)

period1 = (tTwo1 - tOne1) / number1
period2 = (tTwo2 - tOne2) / number2
period1E = (tTwo1E - tOne1E) / number1
period2E = (tTwo2E - tOne2E) / number2

with open('build/tex/i_exp_raw_87.tex', 'w') as f:
    f.write(
        table([
                r'$t_\text{87} / \si{\milli\second}$',
                r'$U_\text{87} / \si{\volt}$'
                ], [
                    expX1E * 1e3,
                    expY1E
                ])
            )

with open('build/tex/i_exp_raw_85.tex', 'w') as f:
    f.write(
        table([
                r'$t_\text{85} / \si{\milli\second}$',
                r'$U_\text{85} / \si{\volt}$'
                ], [
                    expX2E * 1e3,
                    expY2E
                ])
            )

x = np.linspace(expX1[0]*1e3, expX1[-1]*1e3, 10**5)
plt.plot(
    expX1*1e3,
    expY1,
    label=r'Messdaten'
    )
plt.title('Rubidium 87')
plt.legend(loc='best')
plt.xlabel(r't / $ms$')
plt.ylabel(r'U / $V$')
plt.tight_layout()
plt.grid()
plt.xlim([expX1[0]*1e3*99/100, expX1[-1]*1e3*101/100])
plt.savefig('build/plots/fit_E_87.pdf')
plt.clf()

x = np.linspace(expX2[0]*1e3, expX2[-1]*1e3, 10**5)
plt.plot(
    expX2*1e3,
    expY2,
    label=r'Messdaten'
    )
plt.title('Rubidium 85')
plt.legend(loc='best')
plt.xlabel(r't / $ms$')
plt.ylabel(r'U / $V$')
plt.tight_layout()
plt.grid()
plt.xlim([expX2[0]*1e3*99/100, expX2[-1]*1e3*101/100])
plt.savefig('build/plots/fit_E_85.pdf')
plt.clf()

summer1 = sum(expY1E[-500:])
average1 = summer1 / len(expY1E[-500:])

summer2 = sum(expY2E[-500:])
average2 = summer2 / len(expY2E[-500:])

poptL1, pcovL1 = curve_fit(
    func_linear, expX1[600:1226]*1e3,
    -np.log(-expY1[600:1226] + unp.nominal_values(average1)),
    p0=(0, 0)
    )

poptL2, pcovL2 = curve_fit(
    func_linear, expX2[810:1426]*1e3,
    -np.log(-expY2[810:1426] + unp.nominal_values(average2)),
    p0=(0, 0)
    )

x = np.linspace(expX1[600]*1e3*0.9, expX1[1226]*1e3*1.1, 10**5)
plt.plot(
    expX1*1e3,
    -np.log(abs(-expY1 + unp.nominal_values(average1))),
    label=r'Messdaten'
    )
plt.plot(x, func_linear(x, *poptL1), label=r'Regressionskurve $c(t)$')
plt.title('Rubidium 87')
plt.legend(loc='best')
plt.xlabel(r't / $ms$')
plt.ylabel(
    r'$\ln$(U - ' + str(np.round(unp.nominal_values(average1), 4)) + r'$\,V$ / $V$)'
    )
plt.xlim([expX1[0]*1e3*99/100, expX1[-1]*1e3*101/100])
plt.tight_layout()
plt.grid()
plt.savefig('build/plots/fit_L_87.pdf')
plt.clf()

x = np.linspace(expX2[810]*1e3*0.9, expX2[1426]*1e3*1.1, 10**5)
plt.plot(
    expX2*1e3,
    -np.log(abs(-expY2 + unp.nominal_values(average2))),
    label=r'Messdaten'
    )
plt.plot(x, func_linear(x, *poptL2), label=r'Regressionskurve $d(t)$')
plt.title('Rubidium 85')
plt.legend(loc='best')
plt.xlabel(r't / $ms$')
plt.ylabel(
    r'$\ln$(U - ' + str(np.round(unp.nominal_values(average2), 4)) + r'$\,V$ / $V$)'
    )
plt.xlim([expX2[0]*1e3*99/100, expX2[-1]*1e3*101/100])
plt.tight_layout()
plt.grid()
plt.savefig('build/plots/fit_L_85.pdf')
plt.clf()

# Print
nameArray = np.array(['87', '85'])
mLArray = unp.uarray([poptL1[0], poptL2[0]], [pcovL1[0][0], abs(pcovL2[0][0])])
bLArray = unp.uarray([poptL1[1], poptL2[1]], [pcovL1[1][1], abs(pcovL2[1][1])])

with open('build/tex/i_average_87.tex', 'w') as f:
    f.write(r'\SI{{{:L}}}{{\volt}}'.format(average1))

with open('build/tex/i_average_85.tex', 'w') as f:
    f.write(r'\SI{{{:L}}}{{\volt}}'.format(average2))

with open('build/tex/i_fit_start_87.tex', 'w') as f:
    f.write(r'\SI{{{:.4f}}}{{\milli\second}}'.format(expX1[600] * 1e3))

with open('build/tex/i_fit_start_85.tex', 'w') as f:
    f.write(r'\SI{{{:.4f}}}{{\milli\second}}'.format(expX2[810] * 1e3))

with open('build/tex/i_fit_end_87.tex', 'w') as f:
    f.write(r'\SI{{{:.4f}}}{{\milli\second}}'.format(expX1[1226] * 1e3))

with open('build/tex/i_fit_end_85.tex', 'w') as f:
    f.write(r'\SI{{{:.4f}}}{{\milli\second}}'.format(expX2[1426] * 1e3))

with open('build/tex/i_fit_m_87.tex', 'w') as f:
    f.write(r'\SI{{{:.4f}}}{{1\per\milli\second}}'.format(poptL1[0]))

with open('build/tex/i_fit_m_85.tex', 'w') as f:
    f.write(r'\SI{{{:.4f}}}{{1\per\milli\second}}'.format(poptL2[0]))

with open('build/tex/i_fit_tau_87.tex', 'w') as f:
    f.write(r'\SI{{{:.4f}}}{{1\per\milli\second}}'.format(-poptL1[0]))

with open('build/tex/i_fit_tau_85.tex', 'w') as f:
    f.write(r'\SI{{{:.4f}}}{{1\per\milli\second}}'.format(-poptL2[0]))

with open('build/tex/i_fit_b_87.tex', 'w') as f:
    f.write(r'\SI{{{:L}}}{{}}'.format(bLArray[0]))

with open('build/tex/i_fit_b_85.tex', 'w') as f:
    f.write(r'\SI{{{:L}}}{{}}'.format(bLArray[1]))

# Period

with open('build/tex/t_87.tex', 'w') as f:
    f.write(
        table([
                r'$V_\text{87} / \si{\volt}$',
                r'$t_\text{1, 87} / \si{\milli\second}$',
                r'$t_\text{2, 87} / \si{\milli\second}$',
                r'Perioden$_\text{87}$',
                r'$T_\text{87} / \si{\milli\second}$'
                ], [
                    V1,
                    tOne1E*10**3,
                    tTwo1E*10**3,
                    number1,
                    period1E*10**3
                ])
            )

with open('build/tex/t_85.tex', 'w') as f:
    f.write(
        table([
                r'$V_\text{85} / \si{\volt}$',
                r'$t_\text{1, 85} / \si{\milli\second}$',
                r'$t_\text{2, 85} / \si{\milli\second}$',
                r'Perioden$_\text{85}$',
                r'$T_\text{85} / \si{\milli\second}$'
                ], [
                    V2,
                    tOne2E*10**3,
                    tTwo2E*10**3,
                    number2,
                    period2E*10**3
                ])
            )

poptT1, pcovT1 = curve_fit(
    func_hyp, V1, period1*1e3, p0=(0, 0, 0)
    )

poptT2, pcovT2 = curve_fit(
    func_hyp, V2, period2*1e3, p0=(0, 0, 0)
    )

with open('build/tex/i_fit_T_a_87.tex', 'w') as f:
    f.write(
        r'\SI{{{:L}}}{{\micro\second}}'.format(
            ufloat(poptT1[0], pcovT1[0][0]) * 1e3
            )
        )

with open('build/tex/i_fit_T_a_85.tex', 'w') as f:
    f.write(
        r'\SI{{{:L}}}{{\micro\second}}'.format(
            ufloat(poptT2[0], pcovT2[0][0]) * 1e3
            )
        )

with open('build/tex/i_fit_T_b_87.tex', 'w') as f:
    f.write(
        r'\SI{{{:L}}}{{\volt\milli\second}}'.format(
            ufloat(poptT1[1], pcovT1[1][1])
            )
        )

with open('build/tex/i_fit_T_b_85.tex', 'w') as f:
    f.write(
        r'\SI{{{:L}}}{{\volt\milli\second}}'.format(
            ufloat(poptT2[1], pcovT2[1][1])
            )
        )

with open('build/tex/i_fit_T_c_87.tex', 'w') as f:
    f.write(
        r'\SI{{{:L}}}{{\milli\volt}}'.format(
            ufloat(poptT1[2], pcovT1[2][2]) * 1e3
            )
        )

with open('build/tex/i_fit_T_c_85.tex', 'w') as f:
    f.write(
        r'\SI{{{:L}}}{{\milli\volt}}'.format(
            ufloat(poptT2[2], pcovT2[2][2]) * 1e3
            )
        )

with open('build/tex/i_fit_T_quot.tex', 'w') as f:
    f.write(
        r'\SI{{{:L}}}{{}}'.format(
            ufloat(poptT1[1], pcovT1[1][1]) / ufloat(poptT2[1], pcovT2[1][1])
            )
        )

x = np.linspace(V1[0]*0.95, V1[-1]*1.05, 10**5)
plt.plot(x, func_hyp(x, *poptT1), label=r'Regressionskurve $e(U)$')
plt.errorbar(
    V1, period1, yerr=unp.std_devs(period1E), fmt='k.', markersize=1
    )
plt.plot(V1, period1*1e3, 'rx', label=r'Messwerte')
plt.title('Rubidium 87')
plt.legend(loc='best')
plt.xlabel(r'U / $V$')
plt.ylabel(r'T / $ms$')
plt.tight_layout()
plt.grid()
plt.xlim((V1[0]*0.9, V1[-1]*1.1))
plt.savefig('build/plots/fit_T_87.pdf')
plt.clf()

x = np.linspace(V2[0]*0.95, V2[-1]*1.05, 10**5)
plt.plot(x, func_hyp(x, *poptT2), label=r'Regressionskurve $f(U)$')
plt.errorbar(
    V2, period2, yerr=unp.std_devs(period2E), fmt='k.', markersize=1
    )
plt.plot(V1, period2*1e3, 'rx', label=r'Messwerte')
plt.title('Rubidium 85')
plt.legend(loc='best')
plt.xlabel(r'U / $V$')
plt.ylabel(r'T / $ms$')
plt.tight_layout()
plt.xlim((V2[0]*0.9, V2[-1]*1.1))
plt.grid()
plt.savefig('build/plots/fit_T_85.pdf')
plt.clf()
