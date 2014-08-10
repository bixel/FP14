#! /usr/bin/env python3.3
import numpy as np
from uncertainties import ufloat, umath

# Raumwinkelanteil Ω,
# Quellenabstand a,
# Detektorradius r
a = ufloat(7.25, 0.05)
r = 2.25
omega = 2 * np.pi * (1 - a / umath.sqrt(a * a + r * r))
print('Ω / 4π = {}'.format((omega / 2 * np.pi)))
