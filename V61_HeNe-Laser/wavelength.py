#! /usr/bin/env python3

import numpy as np
from codecs import open
from uncertainties import ufloat
import uncertainties.unumpy as unp

# slot
d = ufloat(645, 5)
s = ufloat(0.05, 0.01)
offset = ufloat(5.8, 0.58)

left, right = np.genfromtxt('data/slot.txt', unpack=True)

left -= offset.n
right -= offset.n


def lam(x, n=1, s=s, d=d):
    alpha = unp.arctan(x/d)
    return s / n * unp.sin(alpha)

lambdas_left = unp.uarray([lam(x, n=n+1).n for n, x in enumerate(left)],
                          [lam(x, n=n+1).s for n, x in enumerate(left)])
lambdas_right = unp.uarray([lam(x, n=n+1).n for n, x in enumerate(right)],
                           [lam(x, n=n+1).s for n, x in enumerate(right)])
lambdas = np.append(np.abs(lambdas_left), np.abs(lambdas_right))

(open('build/lambda_slot.tex', 'w', 'utf-8')
 .write(r'$\overline{{\lambda}}_\text{{Spalt}}'
        r'= \SI{{{:.0f}+-{:.0f}}}{{\nano\meter}}$'
        .format(unp.nominal_values(lambdas).mean() * 1e6,
                unp.std_devs(lambdas).mean() * 1e6)))

for ls, xs, name in [[lambdas_right, right, 'mm_r'],
                     [lambdas_left, left, 'mm_l']]:
    f = open('build/lambdas_slot_' + name + '.tex', 'w', 'utf-8')
    for l, x in zip(ls, xs):
        f.write((r'{:.1f} & {} \\' '\n').format(x, l*1e6))
    f.close()


# grid
s = ufloat(0.01, 0.0)
d = ufloat(980, 5)

xs1 = 10 * np.genfromtxt('data/grid1.txt', unpack=True)
xs2_right = np.genfromtxt('data/grid2_right.txt', unpack=True)
xs2_left = np.genfromtxt('data/grid2_left.txt', unpack=True)

offset = 5.0
xs2_right -= offset
xs2_left -= offset

lambdas1 = np.array([lam(x, n=n+1, s=s, d=d) for n, x in enumerate(xs1)])

d = ufloat(132, 5)
lambdas2_right = np.array([lam(x, n=n+1, s=s, d=d)
                           for n, x in enumerate(np.abs(xs2_right))])
lambdas2_left = np.array([lam(x, n=n+1, s=s, d=d)
                          for n, x in enumerate(np.abs(xs2_left))])

lambdas = np.append(lambdas1, lambdas2_right)
lambdas = np.append(lambdas, lambdas2_left)

(open('build/lambda_grid.tex', 'w', 'utf-8')
 .write(r'$\overline{{\lambda}}_\text{{Gitter}}'
        r'= \SI{{{:.0f}+-{:.0f}}}{{\nano\meter}}$'
        .format(unp.nominal_values(lambdas).mean() * 1e6,
                unp.std_devs(lambdas).mean() * 1e6)))

for ls, xs, name in [[lambdas1, xs1, 'cm'],
                     [lambdas2_right, xs2_right, 'mm_r'],
                     [lambdas2_left, xs2_left, 'mm_l']]:
    f = open('build/lambdas_grid_' + name + '.tex', 'w', 'utf-8')
    for l, x in zip(ls, xs):
        f.write((r'{} & {} \\' '\n').format(x, l*1e6))
    f.close()
