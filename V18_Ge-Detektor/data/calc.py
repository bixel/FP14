#! /usr/bin/env python3.3
import numpy as np
from uncertainties import ufloat, umath

# Raumwinkelanteil Ω,
# Quellenabstand a,
# Detektorradius r
d = ufloat(7.25, 0.05)
a = d + 1.5
r = 2.25
omega = 2 * np.pi * (1 - a / umath.sqrt(a * a + r * r))
print('Ω / 4π = {}%'.format(100 * (omega / (4 * np.pi))))

# Aktivität A
A0 = ufloat(4130, 60)
dt = 5006
tau = ufloat(4943, 5)
A_ges = A0 * umath.exp(-dt/tau)
A = A_ges * omega / (4 * np.pi)
print('A_ges = {}'.format(A_ges))
print('A = {}'.format(A))

# Thoretische Auflösungen
Eel = 2.9e-3
Egamma = ufloat(661.623, 0.00589295)
E12T = umath.sqrt(8 * umath.log(2)) * umath.sqrt(0.1 * Egamma * Eel)
print('E12T = {}'.format(E12T))

# Zehntelwertsbreiten
E12 = ufloat(1.344, 0.005)
E110 = ufloat(2.449, 0.009)
print(
    'E12      = {}\n'
    'E110     = {}\n'
    'E110/E12 = {}'.format(
        E12,
        E110,
        E110 / E12
    )
)

# Theoretische Kanten
me = 511.
c = 1.0
eps = Egamma / (me * c*c)
Ec = Egamma * 2 * eps / (1 + 2 * eps)
Er = Egamma * 1 / (1 + 2 * eps)
print(
    'Ec = {}\n'
    'Er = {}'.format(
        Ec,
        Er
    )
)
