#! /usr/bin/env python3.3

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as cfit
from scipy.integrate import simps
# import peakdetect


def sigma_delta(pos, arr, fit_width=30, plot=''):
    """ Try to fit a Gauss to the given datapoints and return the integral of
        the function, aswell as the width sigma.

        @return the integral (simpson integration) of the gauss function from
                -fit_width to fit_width
    """
    gauss = lambda x, A, mu, sigma, B: B + A * np.exp(-(x-mu)**2/(2.*sigma**2))
    data_xs = np.arange(-fit_width, fit_width)
    data_ys = arr[pos - fit_width:pos + fit_width]
    init_values = [1., 0., 1., 0.]
    coeff, var_matrix = cfit(gauss, data_xs, data_ys, p0=init_values)
    xs = np.linspace(-fit_width, fit_width, 1000)
    if plot != '':
        plt.plot(xs, gauss(xs, coeff[0], coeff[1], coeff[2], coeff[3]))
        plt.errorbar(data_xs, data_ys, yerr=np.sqrt(data_ys), fmt='.')
        plt.ylim(ymin=-5)
        plt.xlim(-fit_width, fit_width)
        plt.savefig('{}-{}.pdf'.format(plot, pos))
        plt.clf()
    # return the integral = number of events in peak (without offset)
    return simps(gauss(xs, coeff[0], coeff[1], coeff[2], 0), xs)

# Read Data
europium_distribution = np.loadtxt("Europium.txt", unpack=True)
eur_proof_dist = np.loadtxt("Europium_proof.txt", unpack=True)
background_dist = np.loadtxt("Leermessung.txt", unpack=True)

# literatur Eu152-Spectrum with [Energy[keV], Probability]
eu_spectrum = np.array(
    [[121.78, 0.286],
     [244.7,  0.76],
     [344.3,  0.265],
     [411.12, 0.022],
     [443.96, 0.031],
     [778.9,  0.129],
     [964.08, 0.146],
     [1085.9, 0.102],
     [1112.1, 0.136],
     [1408.0, 0.210]]
)

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

plt.savefig("01_comp.pdf")
plt.clf()

# Remove Background and select maxima
bkg_time = 68580.0
eur_time = 2964.0
maxima_figure = plt.subplot(111)
europium_distribution -= (eur_time / bkg_time) * background_dist

# Own try to detect the peaks
n_max = len(eu_spectrum)

# define array for maxima with index, value of the distribution
# of the maximum
maxima = np.zeros((n_max, 2))
no_max_dist = np.copy(europium_distribution)
# no_max_dist[1200:1400] = 0
no_max_dist[:350] = 0
no_max_dist[400:600] = 0
for index in np.arange(0, n_max):
    max_val_index = np.argmax(no_max_dist)
    maxima[index] = max_val_index, europium_distribution[max_val_index]
    no_max_dist[max_val_index - 10:max_val_index + 10] = 0

print(maxima)

ax = plt.axes()
for maximum in maxima:
    ax.arrow(maximum[0], maximum[1] + 150,
             0, -100,
             head_width=20.0, head_length=20.0)
plt.plot(x_values, europium_distribution[:len(x_values)])
# plt.plot(x_values, (eur_time / bkg_time) * background_dist[:len(x_values)])
plt.title("Detektion der Maxima")
plt.xlabel("Kanal")
plt.ylabel("Anz. Ereignisse")
plt.ylim(0, 4200)

plt.savefig("02_maxima.pdf")
plt.clf()

# Calibrate Energy

# get the channels for calibration
channels = np.sort(maxima[:, 0])

# define and fit linear calibration function
calibration_function = lambda x, m, b: m*x + b
coeff, var = cfit(calibration_function, channels, eu_spectrum[:, 0])
print(
    'Calibration\n===========\n'
    'm = {:.8f}±{:.8f}\n'
    'b = {:.4f}±{:.4f}\n'.format(
        coeff[0],
        np.sqrt(var[0][0]),
        coeff[1],
        np.sqrt(var[1][1])
    )
)

calibration_fig = plt.subplot(111)
x_values = np.arange(0, 4500)
plt.plot(x_values, calibration_function(x_values, coeff[0], coeff[1]))
plt.errorbar(channels, eu_spectrum[:, 0], 0, 0, "ko")
plt.xlabel("Kanal")
plt.ylabel("Energie")
plt.title("Kalibrationsfit")
plt.savefig("03_calibration.pdf")
plt.clf()

calibrated_eu_fig = plt.subplot(111)
plt.plot(calibration_function(x_values, coeff[0], coeff[1]),
         europium_distribution[:len(x_values)])
plt.xlabel("Energie")
plt.ylabel("Anz. Ereignisse")
plt.ylim(0, 4500)
plt.xlim(0, 1500)
plt.savefig("04_eudist_calibrated.pdf")
plt.clf()

# store the integrated number of events for the peaks
europium_events = []
activity = 23.6
grand_total = activity * eur_time
for maximum in maxima:
    europium_events.append(
        sigma_delta(
            maximum[0],
            europium_distribution,
            plot='maximum_fit'
        )
    )

efficiencies = europium_events / (grand_total * eu_spectrum[:,1])
eff_function = lambda x, a, b, c, d: a * np.power(x - b, c) + d
# eff_function = lambda x, a, b: a / x + b
# eff_function = lambda x, a, b, c: a * np.exp(-x / b) + c
print(eu_spectrum[:,0], efficiencies)
coeff, var = cfit(eff_function, eu_spectrum[:, 0][np.array([0,5,6,7,8,9])], efficiencies[np.array([0,5,6,7,8,9])])
print(
    'Efficiency\n==========\n'
    'a = {:g}±{:g}\n'
    'b = {:g}±{:g}\n'
    'c = {:g}±{:g}\n'
    'd = {:g}±{:g}'.format(
        coeff[0], np.sqrt(var[0][0]),
        coeff[1], np.sqrt(var[1][1]),
        coeff[2], np.sqrt(var[2][2]),
        coeff[3], np.sqrt(var[3][3]),
    )
)
xs = np.arange(50, 1600)
plt.plot(xs, eff_function(xs, coeff[0], coeff[1], coeff[2], coeff[3]))
plt.plot(eu_spectrum[:, 0], efficiencies, 'bo')
plt.xlabel('Energie')
plt.ylabel('Effizienz')
plt.savefig('05_efficiencies.pdf')
plt.clf()
