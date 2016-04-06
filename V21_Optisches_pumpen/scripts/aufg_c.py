from codecs import open
from textable import table
import numpy as np
import matplotlib.pylab as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from uncertainties import unumpy as unp

MU_NULL = 4 * np.pi * 10**-7
E_NULL = 1.6021766208 * 10**-19
MASS_NULL = 9.10938356 * 10**-31
PLANCK_H = 6.626070040 * 10**-34
MU_B = 9.274009994 * 10**-24
DELTA_87 = 4.53 * 10**-24
DELTA_85 = 2.01 * 10**-24

R_SWEEP = 16.39 * 10**-2
N_SWEEP = 11
R_VERT = 11.735 * 10**-2
N_VERT = 20
R_HOR = 15.79 * 10**-2
N_HOR = 154
T = 50 + 273.13

S = 1/2
L = 0
J = 1/2

DELTA_I = 0.5 * 10**-3
DELTA_NU = 0.5

peak1 = 6
peak2 = 11
DELTA_PEAK1 = 0.5
DELTA_PEAK2 = 0.5

peak1 = ufloat(peak1, DELTA_PEAK1)
peak2 = ufloat(peak2, DELTA_PEAK2)


def _calc_b(inputData, N, R):
    B = MU_NULL * 8 * inputData * N / (np.sqrt(125) * R)
    return B


def _calc_average(inputData):
    sumNumbers = 0
    for line in inputData:
        sumNumbers += line
    return sumNumbers / len(inputData)


def func_linear(x, a, b):
    return a*x + b


def _calc_g(m):
    g = 4*np.pi*MASS_NULL / (m*E_NULL)
    return g


def _calc_square(inputB, inputG, deltaJ, mList):
    squareList = []
    singleList = []
    for m in mList:
        deltaValOne = inputG * MU_B * unp.nominal_values(inputB)
        deltaValTwo = (1 - 2*m) / deltaJ
        squareList.append(deltaValOne**2 * deltaValTwo)
        singleList.append(deltaValOne)
    return np.array(singleList), np.array(squareList)


def _calc_gj():
    gjz = 3.0023*J*(J+1) + 1.0023*(S*(S+1) - L*(L+1))
    gjn = 2*J*(J+1)
    return np.float64(gjz / gjn)


def _calc_i(gj, gf):
    I1 = -(2 - gj/(2*gf)) / 2
    I2 = I1**2 - 3/4*(1 - gj/gf)
    I2 = np.sqrt(I2)
    return I1 + I2


def _calc_ratio():
    peak2Val = (1 + peak1/peak2)**-1
    peak1Val = 1 - peak2Val
    return peak1Val, peak2Val


nuRaw, sweep0Raw, hor0Raw, sweep1Raw, hor1Raw, sweep2Raw, \
    hor2Raw = np.genfromtxt('data/aufg_c.dat', unpack=True)

nuArr = unp.uarray(nuRaw * 10**3, DELTA_NU)
sweep0Raw = unp.uarray(sweep0Raw, DELTA_I)
sweep1Raw = unp.uarray(sweep1Raw, DELTA_I)
sweep2Raw = unp.uarray(sweep2Raw, DELTA_I)
hor0Raw = unp.uarray(hor0Raw, DELTA_I)
hor1Raw = unp.uarray(hor1Raw, DELTA_I)
hor2Raw = unp.uarray(hor2Raw, DELTA_I)
with open('build/tex/c_data_raw_0.tex', 'w') as f:
    f.write(
        table(
            [
                r'$\nu / \si{\kilo\hertz}$',
                r'$I_\text{0,sweep} / \si{\milli\ampere}$',
                r'$I_\text{0,hor} / \si{\milli\ampere}$'
            ], [
                nuArr*1e-3,
                sweep0Raw*1e3,
                hor0Raw*1e3
            ]
            )
        )

with open('build/tex/c_data_raw_87.tex', 'w') as f:
    f.write(
        table(
            [
                r'$\nu / \si{\kilo\hertz}$',
                r'$I_\text{87,sweep} / \si{\milli\ampere}$',
                r'$I_\text{87,hor} / \si{\milli\ampere}$'
            ], [
                nuArr*1e-3,
                sweep1Raw*1e3,
                hor1Raw*1e3,
            ]
            )
        )

with open('build/tex/c_data_raw_85.tex', 'w') as f:
    f.write(
        table(
            [
                r'$\nu / \si{\kilo\hertz}$',
                r'$I_\text{85,sweep} / \si{\milli\ampere}$',
                r'$I_\text{85,hor} / \si{\milli\ampere}$'
            ], [
                nuArr*1e-3,
                sweep2Raw*1e3,
                hor2Raw*1e3
            ]
            )
        )

sweep0B = _calc_b(sweep0Raw, N_SWEEP, R_SWEEP)
sweep1B = _calc_b(sweep1Raw, N_SWEEP, R_SWEEP)
sweep2B = _calc_b(sweep2Raw, N_SWEEP, R_SWEEP)

hor0B = _calc_b(hor0Raw*3, N_HOR, R_HOR)
hor1B = _calc_b(hor1Raw*3, N_HOR, R_HOR)
hor2B = _calc_b(hor2Raw*3, N_HOR, R_HOR)

b0Ges = hor0B + sweep0B
b1Ges = hor1B + sweep1B
b2Ges = hor2B + sweep2B

with open('build/tex/c_data_b_0.tex', 'w') as f:
    f.write(
        table(
            [
                r'$\nu / \si{\kilo\hertz}$',
                r'$B_\text{0,ges} / \si{\micro\tesla}$',
                r'$B_\text{0,sweep} / \si{\micro\tesla}$',
                r'$B_\text{0,hor} / \si{\micro\tesla}$'
            ], [
                nuArr*1e-3,
                b0Ges*1e6,
                hor0B*1e6,
                sweep0B*1e6
            ])
        )

with open('build/tex/c_data_b_87.tex', 'w') as f:
    f.write(
        table(
            [
                r'$\nu / \si{\kilo\hertz}$',
                r'$B_\text{87,ges} / \si{\micro\tesla}$',
                r'$B_\text{87,sweep} / \si{\micro\tesla}$',
                r'$B_\text{87,hor} / \si{\micro\tesla}$'
            ], [
                nuArr*1e-3,
                b1Ges*1e6,
                hor1B*1e6,
                sweep1B*1e6
            ])
        )

with open('build/tex/c_data_b_85.tex', 'w') as f:
    f.write(
        table(
            [
                r'$\nu / \si{\kilo\hertz}$',
                r'$B_\text{85,ges} / \si{\micro\tesla}$',
                r'$B_\text{85,sweep} / \si{\micro\tesla}$',
                r'$B_\text{85,hor} / \si{\micro\tesla}$'
            ], [
                nuArr*1e-3,
                b2Ges*1e6,
                hor2B*1e6,
                sweep2B*1e6
            ])
        )

bVert = _calc_b(ufloat(0.26, DELTA_I), N_VERT, R_VERT)
bHor = _calc_average(b0Ges)

with open('build/tex/c_vert.tex', 'w') as f:
    f.write(r'\SI{{{:L}}}{{\micro\tesla}}'.format(bVert*1e6))

with open('build/tex/c_data_erd.tex', 'w') as f:
    f.write(
        table(
            [
                r'$B_\text{Vert} / \si{\micro\tesla}$',
                r'$B_\text{Hor} / \si{\micro\tesla}$'
            ], [
                bVert*1e6,
                bHor*1e6
            ])
        )

popt1, pcov1 = curve_fit(
    func_linear, unp.nominal_values(nuArr),
    unp.nominal_values(b1Ges)
    )
popt2, pcov2 = curve_fit(
    func_linear, unp.nominal_values(nuArr),
    unp.nominal_values(b2Ges)
    )

with open('build/tex/c_fit_m_87.tex', 'w') as f:
    f.write(r'\SI{%1.4e}{\micro\tesla\per\mega\per\hertz}' % (popt1[0]))
with open('build/tex/c_fit_b_87.tex', 'w') as f:
    f.write(r'\SI{%1.4e}{\micro\tesla\per\mega\per\hertz}' % (popt1[1]))
with open('build/tex/c_fit_m_85.tex', 'w') as f:
    f.write(r'\SI{%1.4e}{\micro\tesla\per\mega\per\hertz}' % (popt2[0]))
with open('build/tex/c_fit_b_85.tex', 'w') as f:
    f.write(r'\SI{%1.4e}{\micro\tesla\per\mega\per\hertz}' % (popt2[1]))

gf1 = _calc_g(popt1[0])
gf2 = _calc_g(popt2[0])
gj = _calc_gj()
i1 = _calc_i(gj, gf1)
i2 = _calc_i(gj, gf2)

with open('build/tex/c_I_87.tex', 'w') as f:
    f.write(r'\SI{{{:.4f}}}{{}}'.format(i1))

with open('build/tex/c_I_85.tex', 'w') as f:
    f.write(r'\SI{{{:.4f}}}{{}}'.format(i2))


with open('build/tex/c_data_gi.tex', 'w') as f:
    f.write(
        table(
            [
                r'$g_\text{f,87}$',
                r'$g_\text{f,85}$',
                r'$g_\text{j}$',
                r'$I_\text{87}$',
                r'$I_\text{85}$'
            ], [
                gf1.round(4),
                gf2.round(4),
                gj.round(4),
                i1.round(4),
                i2.round(4)
            ]
            )
        )


mList = np.array([-2, -1, 0, 1, 2])

square1 = np.array(_calc_square(b1Ges, gf1, DELTA_87, mList))
square2 = np.array(_calc_square(b2Ges, gf2, DELTA_85, mList))

deltaSquare1 = square1[1] / square1[0]
deltaSquare2 = square2[1] / square2[0]

mArr = np.array([-2, -1, 0, 1, 2])
mArray = np.empty(len(deltaSquare1[0]))

for m in range(len(mArr)):
    mArray.fill(m - 2)
    with open('build/tex/c_data_square_87_' + str(m) + '.tex', 'w') as f:
        f.write(
            table([
                    r'M$_\text{F}$',
                    r'B$_\text{Ges., 87} / \si{\micro\tesla}$',
                    r'U$_\text{1., 87} / 10^{-30} \si{\joule}$',
                    r'U$_\text{2., 87} / 10^{-33} \si{\joule}$',
                    r'$\frac{\text{U}_\text{2., 87}}{\text{U}_\text{1., 87}} / 10^{-3}$'
                    ], [
                        mArray,
                        b1Ges * 1e6,
                        np.round(square1[0][m] * 1e30, 4),
                        np.round(square1[1][m] * 1e33, 4),
                        np.round(deltaSquare1[m] * 1e3, 4)
                    ])
                )

for m in range(len(mArr)):
    mArray.fill(m-2)
    with open('build/tex/c_data_square_85_' + str(m) + '.tex', 'w') as f:
        f.write(
            table([
                    r'M$_\text{F}$',
                    r'B$_\text{Ges., 85} / \si{\micro\tesla}$',
                    r'U$_\text{1., 85} / 10^{-30} \si{\joule}$',
                    r'U$_\text{2., 85} / 10^{-33} \si{\joule}$',
                    r'$\frac{\text{U}_\text{2., 85}}{\text{U}_\text{1., 85}} / 10^{-3}$'
                    ], [
                        mArray,
                        b2Ges * 1e6,
                        np.round(square2[0][m] * 1e30, 4),
                        np.round(square2[1][m] * 1e33, 4),
                        np.round(deltaSquare2[m] * 1e3, 4)
                    ])
                )

with open('build/tex/c_data_square_87.tex', 'w') as f:
    m = np.array([[m]*10 for m in mList])
    b = np.array([[b1Ges]*5])
    f.write(
        table([
                r'M$_\text{F}$',
                r'B$_\text{Ges., 87} / \si{\micro\tesla}$',
                r'U$_\text{1., 87} / 10^{-27} \si{\joule}$',
                r'U$_\text{2., 87} / 10^{-33} \si{\joule}$',
                r'$\frac{\text{U}_\text{2., 87}}{\text{U}_\text{1., 87}} / 10^{-3}$'
                ], [
                    m.flatten(),
                    b[0].flatten() * 1e6,
                    np.round(square1[0].flatten() * 1e27, 4),
                    np.round(square1[1].flatten() * 1e33, 4),
                    np.round(deltaSquare1.flatten() * 1e3, 4)
                ])
            )

with open('build/tex/c_data_square_85.tex', 'w') as f:
    m = np.array([[m]*10 for m in mList])
    b = np.array([[b2Ges]*5])
    f.write(
        table([
                r'M$_\text{F}$',
                r'B$_\text{Ges., 85} / \si{\micro\tesla}$',
                r'U$_\text{1., 85} / 10^{-27} \si{\joule}$',
                r'U$_\text{2., 85} / 10^{-33} \si{\joule}$',
                r'$\frac{\text{U}_\text{2., 85}}{\text{U}_\text{1., 85}} / 10^{-3}$'
                ], [
                    m.flatten(),
                    b[0].flatten() * 1e6,
                    np.round(square2[0].flatten() * 1e27, 4),
                    np.round(square2[1].flatten() * 1e33, 4),
                    np.round(deltaSquare2.flatten() * 1e3, 4)
                ])
            )

ratio1, ratio2 = _calc_ratio()

with open('build/tex/c_ratio_87.tex', 'w') as f:
    f.write(r'\num{{{:L}}}'.format(peak1))

with open('build/tex/c_ratio_85.tex', 'w') as f:
    f.write(r'\num{{{:L}}}'.format(peak2))

with open('build/tex/c_data_ratio.tex', 'w') as f:
    f.write(
        table([
                r'Rb$_\text{87} / \si{\percent}$',
                r'Rb$_\text{85} / \si{\percent}$'
                ], [
                    ratio1*100,
                    ratio2*100
                ])
            )

popt1, pcov1 = curve_fit(
    func_linear, unp.nominal_values(nuArr)*1e-6,
    unp.nominal_values(b1Ges)*1e6
    )
popt2, pcov2 = curve_fit(
    func_linear, unp.nominal_values(nuArr)*1e-6,
    unp.nominal_values(b2Ges)*1e6
    )

x = np.linspace(
    unp.nominal_values(nuArr[0])*1e-6,
    unp.nominal_values(nuArr[-1])*1e-6, 10**5
    )
plt.plot(x, func_linear(x, *popt1), label=r'Regressionskurve $a(\nu)$')
plt.errorbar(
    unp.nominal_values(nuArr)*1e-6,
    unp.nominal_values(b1Ges)*1e6,
    yerr=unp.std_devs(b1Ges), fmt='kx',
    label=r'Messwerte'
    )
plt.title('Rubidium 87')
plt.ylabel(r'B / $\mu$T')
plt.xlabel(r'$\nu$ / MHz')
plt.tight_layout()
plt.legend(loc='best')
plt.xlim(
    unp.nominal_values(x[0])*90/100,
    unp.nominal_values(x[-1])*101/100
    )
plt.grid()
plt.savefig('build/plots/c_fit_87.pdf')
plt.clf()

plt.plot(x, func_linear(x, *popt2), label=r'Regressionskurve $b(\nu)$')
plt.errorbar(
    unp.nominal_values(nuArr)*1e-6,
    unp.nominal_values(b2Ges)*1e6,
    yerr=unp.std_devs(b2Ges), fmt='kx',
    label=r'Messwerte'
    )
plt.title('Rubidium 85')
plt.ylabel(r'B / $\mu$T')
plt.xlabel(r'$\nu$ / MHz')
plt.tight_layout()
plt.xlim(
    unp.nominal_values(x[0])*90/100,
    unp.nominal_values(x[-1])*101/100
    )
plt.grid()
plt.legend(loc='best')
plt.savefig('build/plots/c_fit_85.pdf')
plt.clf()
