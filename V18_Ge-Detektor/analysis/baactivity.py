""" Calculate Ba-Activity
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from peakdetect import get_maxima, fit_gauss
from unbinner import unbinned_array
from calibration import E
from efficiency import eff
from uncertainties import (ufloat,
                           umath,
                           unumpy as unp)
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)

def activity_func(x, m, b):
    return m * x + b

if __name__ == '__main__':
    # calibration
    cal_coeff, cal_errs = np.genfromtxt('../build/data/calibration.txt',
            unpack=True)
    ba_dist = np.genfromtxt('../data/Barium.txt', unpack=True)[:5000]
    xs = E(np.arange(0,5000), *cal_coeff)

    # find maxima and sort by energy
    vetos = [[780, 830]]
    maxima = get_maxima(ba_dist, n_max=5, vetos=vetos)
    maxima = np.array(sorted(maxima, key=lambda x: x[0]))

    # fit position and content of peaks
    ba_events = []
    for maximum in maxima:
        fig, ax = plt.subplots(1, 1)
        x, s, I = fit_gauss(xs, ba_dist, xs[maximum[0]], fit_width=3, ax=ax)
        ba_events.append([x, s, I])
        ax.set_xlabel('Energie / keV')
        ax.set_ylabel('Ereignisse')
        fig.savefig('../build/plots/ba_gauss-'+str(maximum[0])+'.pdf')
        fig.clf()
    ba_events = np.array(ba_events)

    # efficiency-parameters
    ba_energies, ba_props = np.genfromtxt('../data/ba_spectrum_lit.txt',
            unpack=True)
    eff_coeff, eff_errs = np.genfromtxt('../build/data/efficiency-params.txt',
            unpack=True)

    xs = ba_props * eff(noms(ba_events[:, 0]), *eff_coeff)
    ys = np.abs(noms(ba_events[:, 2]))
    y_errs = stds(ba_events[:, 2])

    act_coeff, act_covar = curve_fit(activity_func, xs, ys, sigma=y_errs)
    act_errs = np.sqrt(np.diag(act_covar))

    x = np.linspace(0, 0.25)
    plt.plot(x, activity_func(x, *act_coeff), label='Fit')
    plt.errorbar(xs, ys, yerr=y_errs, fmt='+', label='Datenpunkte')
    plt.legend(loc='best')
    plt.xlabel(r'$W \cdot Q$')
    plt.ylabel('Ereignisse')
    plt.savefig('../build/plots/ba_activity.pdf')
    plt.clf()

    np.savetxt('../build/data/ba_activity-params.txt',
            np.array([act_coeff, act_errs]).T)

    ba_time = 3551
    activity = ufloat(act_coeff[0], act_errs[0]) / ba_time / ufloat(0.01575, 0.00017)
    np.savetxt('../build/data/ba_activity.txt',
            np.array([activity.n, activity.s]))


