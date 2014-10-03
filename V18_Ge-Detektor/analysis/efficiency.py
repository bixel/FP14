import numpy as np
from calibration import E
from uncertainties import ufloat, unumpy as unp
from uncertainties.unumpy import nominal_values as noms, std_devs as stds
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def eff(x, a, b, c): #, d):
    return a * np.exp(-b * x) + c
    # return a * np.power(x - b, c) + d


if __name__ == '__main__':
    # load data
    eu_measurement_time = 2970
    eu_energies, eu_probs = np.genfromtxt('../data/eu_spectrum_lit.txt', unpack=True)
    cal_coeff, cal_errs = np.genfromtxt('../build/data/calibration.txt',
            unpack=True)
    mean, mean_err, sigma, sigma_err, content, content_err = np.genfromtxt(
        '../build/data/calibration-maxima.txt', unpack=True)
    peaks = unp.uarray(mean, mean_err)
    # calibrate peak-position
    peaks = E(peaks, *unp.uarray(cal_coeff, cal_errs))
    contents = unp.uarray(content, content_err)

    # calculate acitvity
    volume_part = ufloat(0.01575, 0.00017)
    activity = ufloat(1500, 22)
    expected = volume_part * activity * eu_probs * eu_measurement_time
    efficiency = contents / expected

    # create mask for disabling some data-points
    mask = np.array([0,1,0,0,0,0,0,0,0,0])
    nmask = mask ^ 1
    xs = np.ma.masked_array(noms(peaks), mask)
    ys = np.ma.masked_array(noms(efficiency), mask)
    y_errs = np.ma.masked_array(stds(efficiency), mask)
    dis_xs = np.ma.masked_array(noms(peaks), nmask)
    dis_ys = np.ma.masked_array(noms(efficiency), nmask)
    dis_ys_errs = np.ma.masked_array(stds(efficiency), nmask)

    # p0 = [0.5, 1, -0.1, 0.1]
    p0 = [0.5, 0.01, 0.1]
    coeff, covar = curve_fit(eff, noms(peaks), noms(efficiency), p0=p0, sigma=y_errs)
    errs = np.sqrt(np.diag(covar))
    x = np.linspace(0.1, 1500, 200)
    plt.plot(x, eff(x, *coeff), label='Fit')
    plt.errorbar(xs, ys, yerr=y_errs, fmt='+', label='Datenpunkte')
    plt.errorbar(dis_xs, dis_ys, yerr=dis_ys_errs, fmt='+', label='Nicht berücksichtigt')
    plt.legend(loc='best')
    plt.ylim(0, 0.8)
    plt.xlim(0, 1500)
    plt.xlabel('Energie / keV')
    plt.ylabel('Effizienz')
    plt.savefig('../build/plots/efficiency.pdf')
    plt.clf()

    # store fit parameter
    np.savetxt('../build/data/efficiency-params.txt', np.array([coeff, errs]).T)
    np.savetxt('../build/data/efficiencies.txt',
            np.array([
                noms(expected),
                stds(expected),
                noms(efficiency),
                stds(efficiency)
            ]).T, header='n_exp\texp_err\teff\teff_err', fmt='%5.3f')
