#! /usr/bin/env python3.3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit as cfit
from scipy.integrate import simps
from uncertainties import ufloat, umath
# import peakdetect


def sigma_delta(pos, arr, fit_width=30, plot='', calibration_func=None):
    """ Try to fit a Gauss to the given datapoints and return the integral of
        the function, aswell as the width sigma.

        @return the integral (simpson integration) of the gauss function from
                -fit_width to fit_width
    """
    gauss = lambda x, A, mu, sigma, B: B + A * np.exp(- 0.5 * ((x-mu)/sigma)**2)
    data_xs = np.arange(-fit_width, fit_width)
    data_ys = arr[pos - fit_width:pos + fit_width]
    init_values = [arr[pos], 0., 5., 0.]
    coeff, var_matrix = cfit(gauss, data_xs, data_ys, p0=init_values)
    xs = np.linspace(-fit_width, fit_width, 1000)
    if plot != '':
        plt.plot(xs + pos, gauss(xs, coeff[0], coeff[1], coeff[2], coeff[3]))
        plt.errorbar(data_xs + pos, data_ys, yerr=np.sqrt(data_ys), fmt='.')
        plt.ylim(ymin=-5)
        sigma = coeff[2]
        plt.xlim(pos - 5*sigma, pos + 5*sigma)
        if calibration_func != None:
            xticks, xlabels = plt.xticks()
            xlabels = list(map(lambda x: '%.1f' % x, calibration_func(xticks)))
            plt.xticks(xticks, xlabels)
        plt.ylabel('Anzahl Ereignisse')
        plt.xlabel('Energie [keV]')
        plt.legend(["Gaußscher Fit", "Datenpunkte"], loc='best')
        plt.savefig('{}-{}.pdf'.format(plot, pos))
        plt.clf()
    # return the integral = number of events in peak (without offset)
    return (
        simps(gauss(xs, coeff[0], coeff[1], coeff[2], 0), xs),
        coeff,
        var_matrix
    )

def peaks(dist, n_max=10, vetos = []):
    """ Generate an array with position of maximum value and maximum value
        for given distribution-array.
        Look for n_max maxima.
        vetos defines the ranges excluded from searching

        @return array with [index, value] of the n_max highest maxima
    """
    maxima = np.zeros((n_max, 2))
    no_max_dist = np.copy(dist)
    for veto in vetos:
        no_max_dist[veto[0]:veto[1]] = 0
    for index in np.arange(0, n_max):
        max_val_index = np.argmax(no_max_dist)
        maxima[index] = max_val_index, europium_distribution[max_val_index]
        no_max_dist[max_val_index - 10:max_val_index + 10] = 0
    return maxima

def unbinned_array(binned_array, start=0, end=None, calibration_func=lambda x: x):
    """ Returned an array with 'raw' Data for creating histograms with plt.hist
    """
    raw_data = []
    if end == None:
        end = len(binned_array) - 1
    for x in range(start,end):
        raw_data.extend([calibration_func(x)] * binned_array[x])
    return raw_data

# Read Data
europium_distribution = np.loadtxt("Europium.txt", unpack=True)
eur_proof_dist = np.loadtxt("Europium_proof.txt", unpack=True)
background_dist = np.loadtxt("Leermessung.txt", unpack=True)
caesium_dist = np.loadtxt('Caesium.txt', unpack=True)

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

vetos = np.array(
    [[0, 350],
     [400, 600]]
)
maxima = peaks(europium_distribution, vetos=vetos)

print(peaks(europium_distribution, vetos=[[0,350], [400,600]]))

ax = plt.axes()
# plt.plot(x_values, europium_distribution[:len(x_values)])
# plt.plot(x_values, (eur_time / bkg_time) * background_dist[:len(x_values)])
heights, positions, bars = plt.hist(unbinned_array(europium_distribution, end=4200), bins=200, histtype='stepfilled', edgecolor='none')
for maximum in maxima:
    height = 0
    for h, pos in zip(heights, positions):
        if pos >= maximum[0] - 4200/200:
            height = h
            break
    ax.arrow(maximum[0], height + 500,
             0, -300,
             head_width=60.0, head_length=100.0, edgecolor='black', facecolor='black')
plt.ylim(ymax=15000)
plt.title("Detektion der Maxima")
plt.xlabel("Kanal")
plt.ylabel("Anz. Ereignisse")
plt.savefig("02_maxima.pdf")
plt.clf()

# Calibrate Energy

# get the channels for calibration
channels = np.sort(maxima[:, 0])

# define and fit linear calibration function
calibration_function = lambda x, m, b: m*x + b
calibration_coeff, calibration_var = cfit(calibration_function, channels, eu_spectrum[:, 0])
print(
    'Calibration\n===========\n'
    'm = {:.8f}±{:.8f}\n'
    'b = {:.4f}±{:.4f}\n'.format(
        calibration_coeff[0],
        np.sqrt(calibration_var[0][0]),
        calibration_coeff[1],
        np.sqrt(calibration_var[1][1])
    )
)

def calibrated(x):
    """ Just a little helper to reduce typing
    """
    return calibration_function(x, calibration_coeff[0], calibration_coeff[1])

calibration_fig = plt.subplot(111)
x_values = np.arange(0, 4500)
plt.plot(x_values, calibration_function(x_values, calibration_coeff[0], calibration_coeff[1]))
plt.errorbar(channels, eu_spectrum[:, 0], 0, 0, "k+")
plt.xlabel("Kanal")
plt.ylabel("Energie")
plt.title("Kalibrationsfit")
plt.savefig("03_calibration.pdf")
plt.clf()

calibrated_eu_fig = plt.subplot(111)
plt.hist(unbinned_array(europium_distribution, end=4200, calibration_func=calibrated), edgecolor='none', histtype='stepfilled', bins=200)
plt.xlabel("Energie [keV]")
plt.ylabel("Anz. Ereignisse")
# plt.ylim(0, 4500)
# plt.xlim(0, 1500)
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
        )[0]
    )

efficiencies = europium_events / (grand_total * eu_spectrum[:,1])

for measured, expected, efficiency in zip(europium_events, grand_total * eu_spectrum[:,1], efficiencies):
    print('{:.0f}\t{:.0f}\t{:2.1f}'.format(measured, expected, 100 * efficiency))

# eff_function = lambda x, a, b, c, d: a * np.power(x - b, c) + d
# eff_function = lambda x, a, b: a / x + b
# eff_function = lambda x, a, b, c: a * np.exp(-x / b) + c
eff_function = lambda x, a, b: a * np.exp(-b * x)
# coeff, var = cfit(eff_function, eu_spectrum[:, 0][[0,5,6,7,8,9]], efficiencies[:][[0,5,6,7,8,9]], p0=[1.0, 0.001], maxfev=10000)
q_coeff, q_var = cfit(eff_function, eu_spectrum[:, 0], efficiencies, p0=[1.0, 0.001], maxfev=10000)
print(q_coeff, np.sqrt(q_var))
print(
    'Efficiency\n==========\n'
    'a = {:g}±{:g}\n'
    'b = {:g}±{:g}\n'.format(
        q_coeff[0], np.sqrt(q_var[0][0]),
        q_coeff[1], np.sqrt(q_var[1][1]),
        # q_coeff[2], np.sqrt(q_var[2][2]),
        # q_coeff[3], np.sqrt(q_var[3][3]),
    )
)
xs = np.arange(50, 1600)
plt.plot(xs, eff_function(xs, q_coeff[0], q_coeff[1]))# , q_coeff[2], q_coeff[3]))
plt.plot(eu_spectrum[:, 0], efficiencies, 'b+')
plt.xlabel('Energie')
plt.ylabel('Effizienz')
plt.legend(['Fit Effizienzfunktion', 'Messdaten'], loc='best')
plt.savefig('05_efficiencies.pdf')
plt.clf()

cs_maxima = peaks(caesium_dist, n_max=1)
cs_events, cs_coeff, cs_var = sigma_delta(
    cs_maxima[0, 0],
    caesium_dist,
    plot='caesium_fit',
    calibration_func=calibrated
)

plt.plot(
    calibrated(x_values),
    caesium_dist[:len(x_values)]
)
plt.xlim(0, 800)
plt.xlabel('Energie [keV]')
plt.ylabel('Ereignisse')
plt.savefig('06_caesium.pdf')
plt.clf()

cal_sig = calibrated(1000 + cs_coeff[2]) - calibrated(1000)
sigma = ufloat(cal_sig, np.sqrt(cs_var[2][2]) / cs_coeff[2] * cal_sig)
E12  = 2 * sigma * umath.sqrt(2 * umath.log(2))
E110 = 2 * sigma * umath.sqrt(2 * umath.log(10))

print(
    'Caesium-Photopeak\n=================\n'
    'N          = {:g}±{:g}\n'
    'μ          = {:g}±{:g}keV\n'
    'σ          = {}\n'
    'E_1/2      = {}\n'
    'E_1/10     = {}\n'
    'E12 / E119 = {}'.format(
        cs_events, np.sqrt(cs_events),
        calibrated(cs_maxima[0][0] + cs_coeff[1]), np.sqrt(cs_var[1][1]),
        sigma,
        E12,
        E110,
        E110 / E12
    )
)

compton_peak = peaks(caesium_dist, n_max=1, vetos=[[0,1300],[1600,5000]])
reflex_peak = peaks(caesium_dist, n_max=1, vetos=[[1600,5000]])

me = 511.
c = 1.0
# E = calibrated(compton_peak[0][0])
# eps = E / (me * c*c)
compton_function = lambda x, A, B, E, eps: A * (2 + (E/(x - E))**2 * (1/eps**2 + (x - E)/x - 2/eps * (x - E)/x)) + B

fitx = calibrated(np.arange(800,1350))
fity = caesium_dist[800:1350]
coeff, var = cfit(compton_function, fitx, fity, p0=[1.0, 20.0, 477.0, 0.93])

plt.plot(calibrated(np.arange(500, 1500)), caesium_dist[500:1500])
plt.plot(calibrated(np.arange(500, 1500)), compton_function(calibrated(np.arange(500, 1500)), coeff[0], coeff[1], coeff[2], coeff[3]))
plt.ylim(0, 200)
plt.savefig('c.pdf')
plt.clf()

Ic = simps(compton_function([50, compton_peak[0][0]], coeff[0], coeff[1], coeff[2], coeff[3]), [50, compton_peak[0][0]])
Ic = ufloat(Ic, np.sqrt(Ic))

print(
    'params  = {}'
    'compton = {:g}keV\n'
    'reflex  = {:g}keV\n'
    'Ic      = {}'.format(
        coeff,
        calibrated(compton_peak[0][0]),
        calibrated(reflex_peak[0][0]),
        Ic
    )
)

plt.plot(
    calibrated(x_values),
    caesium_dist[:len(x_values)],
    'k+'
)
plt.plot(np.arange(200,510), compton_function(np.arange(200,510), coeff[0], coeff[1], coeff[2], coeff[3]))
plt.xlim(0, 600)
plt.ylim(0, 180)
plt.axvline(x=calibrated(800), ymax=0.4, color='r', linestyle='--')
plt.axvline(x=calibrated(1350), ymax=0.4, color='r', linestyle='--')
plt.xlabel('Energie [keV]')
plt.ylabel('Ereignisse')
plt.legend(['Datenpunkte', 'Fit'], loc='best')
plt.savefig('06_caesium_zoomed.pdf')
plt.clf()

ba_props = np.array(
    [0.341,
    0.183,
    0.006,
    0.621,
    0.089]
)

ba_distribution = np.loadtxt('Barium.txt', unpack=True)
ba_time = 3551.0
ba_distribution -= ba_time / bkg_time * background_dist
unbinned_barium = unbinned_array(ba_distribution[:1200], calibration_func=calibrated)
plt.hist(unbinned_barium, bins=200, edgecolor='none')
# xs = np.arange(0,1200)
# plt.plot(xs, ba_distribution[:1200])
plt.ylabel('Ereignisse')
plt.xlabel('Energie [keV]')
plt.savefig('07_barium.pdf')
plt.clf()

vetos = [[780, 830]]
ba_maxima = peaks(ba_distribution, n_max=5, vetos=vetos)
ba_maxima = np.array(sorted(ba_maxima, key=lambda x: x[0]))
ba_events = []
for maximum in ba_maxima:
    ba_events.append(
        sigma_delta(
            maximum[0],
            ba_distribution,
            plot='barium-fit',
            calibration_func=calibrated
        )[0]
    )
print('Barium-133\n==========')
for m, e in zip(ba_maxima, ba_events):
    print('E = {}\tN = {}'.format(calibrated(m[0]), e))

omega = 0.01575
ba_activity = lambda x, a: omega * a * x
print(ba_maxima)
fitx = np.array(eff_function(calibrated(ba_maxima[:,0]), q_coeff[0], q_coeff[1])) * ba_props
ba_activity_coeff, ba_activity_var = cfit(ba_activity, fitx, ba_events)
plt.plot(fitx, ba_events, 'k+')
xs = np.linspace(0, 0.2, 100)
plt.plot(xs, ba_activity(xs, ba_activity_coeff[0]))
plt.xlabel('$W \cdot Q$')
plt.ylabel('Peak-Ereignisse')
plt.savefig('07_barium_activity.pdf')
plt.clf()

a = ufloat(ba_activity_coeff[0], np.sqrt(ba_activity_var[0][0]))
print('A_ges = {}'.format(a / ba_time))


# Coal-Brikkets

coal_dist = np.loadtxt('Kohle.txt', unpack=True)
coal_time = 84292.0
coal_dist -= coal_time / bkg_time * background_dist
unbinned_coal = unbinned_array(coal_dist, calibration_func=calibrated)
plt.hist(unbinned_coal, bins=200, edgecolor='none')
plt.xlim(0, 1550)
plt.ylabel('Ereignisse')
plt.xlabel('Energie [keV]')
plt.savefig('08_coal.pdf')
plt.clf()

coal_maxima = peaks(coal_dist, n_max=10)
coal_maxima = np.array(sorted(coal_maxima, key=lambda x: x[0]))
coal_events = []
for maximum in coal_maxima:
    coal_events.append(
        sigma_delta(
            maximum[0],
            coal_dist,
            plot='coal-fit',
            calibration_func=calibrated
        )[0]
    )
print('Wooden Coal\n===========')
for m, e in zip(coal_maxima, coal_events):
    print('E = {}\tN = {}'.format(calibrated(m[0]), e))
