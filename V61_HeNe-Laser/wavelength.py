#! /usr/bin/env python3

import numpy as np
import uncertainties.unumpy as unp
from uncertainties import ufloat
from codecs import open

# slot

d = 645
s = 0.05
offset = 5.8

left, right = np.genfromtxt('data/slot.txt', unpack=True)

left -= offset
right -= offset

def lam(x, n=1, s=s, d=d):
    alpha = np.arctan(x/d)
    return s / n * alpha

lambdas_left = np.array([lam(x, n=n+1) for n, x in enumerate(left)])
lambdas_right = np.array([lam(x, n=n+1) for n, x in enumerate(right)])
lambdas = np.append(np.abs(lambdas_left), np.abs(lambdas_right))

(open('build/lambda_slot.tex', 'w', 'utf-8')
        .write(r'$\overline{{\lambda}}_\text{{Spalt}}'
               r'= \SI{{{:3.0f}+-{:3.0f}}}{{\nano\meter}}$'.format(
                    lambdas.mean() * 1e6, lambdas.std() * 1e6         
                    )
              )
)
