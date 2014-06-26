#! /usr/bin/env python3.3

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as cfit
# import peakdetect


def sigma_delta(pos, arr):
    gauss = lambda x, A, mu, sigma: A * np.exp(-(x-mu)**2/(2.*sigma**2))
    init_values = [1., pos, 1.]
    coeff, var_matrix = cfit(gauss, arr, p0=init_values)


# Read Data
europium_distribution = np.loadtxt("Europium.txt", unpack=True)
eur_proof_dist = np.loadtxt("Europium_proof.txt", unpack=True)
background_dist = np.loadtxt("Leermessung.txt", unpack=True)

# Plot raw Eu-152 Distribution
x_values = np.arange(0, 4200)
compare_figure, (eur_plot, eur_proof_plot) = plt.subplots(2, sharex=True)
eur_plot.plot(x_values, europium_distribution[:len(x_values)])
eur_proof_plot.plot(x_values, eur_proof_dist[:len(x_values)])
eur_proof_plot.set_xlabel('Kanal')
eur_proof_plot.set_ylabel('Kalibrationsmessung')
eur_plot.set_ylabel('Vergleichsmessung')

compare_figure.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in compare_figure.axes[:-1]],
         visible=False)
plt.setp([a.get_yticklabels() for a in compare_figure.axes],
         visible=False)

plt.savefig("01_comp.png")
plt.clf()

# Remove Background and select maxima
bkg_time = 68580.0
eur_time = 2964.0
maxima_figure = plt.subplot(111)
europium_distribution -= (eur_time / bkg_time) * background_dist
# europium_distribution = europium_distribution[400:1100]
# europium_distribution_orig = europium_distribution.copy()
#
# # try to use peakdetect.py
# plt.plot(peakdetect._smooth(europium_distribution))
# plt.plot(europium_distribution_orig - 1000)
# peaks = peakdetect.peakdetect(peakdetect._smooth(europium_distribution,
#                                                  31)[10:],
#                               lookahead=15,
#                               delta=20)
#
# ax = plt.axes()
# for peak in peaks[0]:
#     print peak, europium_distribution[peak[0]]
#     ax.arrow(peak[0], peak[1] + 100,
#              0, -50,
#              head_length=0.1)
#
# maxima = peaks[0]

# own try to detect the peaks

n_max = 8

maxima = np.zeros((n_max, 2))
no_max_dist = np.copy(europium_distribution)
no_max_dist[1200:1400] = 0
for index in np.arange(0, n_max):
    max_val_index = np.argmax(no_max_dist[200:]) + 200
    maxima[index] = max_val_index, europium_distribution[max_val_index]
    no_max_dist[max_val_index - 10:max_val_index + 10] = 0
    # for ignore_index in np.arange(max_val_index - 10, max_val_index + 10):
    #     no_max_dist[ignore_index] = 0

ax = plt.axes()
for maximum in maxima:
    ax.arrow(maximum[0], maximum[1] + 150,
             0, -100,
             head_width=20.0, head_length=20.0)
plt.plot(x_values, europium_distribution[:len(x_values)])
plt.plot(x_values, (eur_time / bkg_time) * background_dist[:len(x_values)])
plt.title("Detektion der Maxima")
plt.xlabel("Kanal")
plt.ylabel("Anz. Ereignisse")
plt.ylim(0, 4200)

plt.savefig("02_maxima.pdf")
plt.clf()

# Calibrate Energy
# literatur Eu152-Spectrum with [Energy[keV], Probability]
eu_spectrum = np.array(
    [[121.78, 0.286],
     [244.7,  0.76],
     [344.3,  0.265],
     [778.9,  0.129],
     [964.08, 0.146],
     [1085.9, 0.102],
     [1112.1, 0.136],
     [1408.0, 0.210]]
)

calibration_function = lambda x, m, b: m*x + b
coeff, var = cfit(calibration_function, eu_spectrum[:, 0], maxima[:, 0])
print(coeff, var)


calibration_fig = plt.subplot(111)
energies = np.arange(0, 1500)
plt.errorbar(eu_spectrum[:, 0], maxima[:, 0], 0, 0, fmt="ro")
plt.plot(energies, calibration_function(energies, coeff[0], coeff[1]))
# plt.xlabel("$E\,[\si{\kilo \electronvolt}]$")
plt.ylabel("Kanal")
plt.title("Kalibrationsfit")
plt.savefig("03_calibration.pdf")
plt.clf()
