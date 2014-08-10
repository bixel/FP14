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
