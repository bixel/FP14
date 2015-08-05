import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat
from uncertainties import unumpy as unp
from scipy.optimize import curve_fit
from codecs import open
from textable import table

U_B = 28.3 / 2
R_1 = 100
R_P = 4.7e3
nu = 1e3

print( U_B * R_1 / R_P )