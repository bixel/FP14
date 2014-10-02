import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import (ufloat,
                           umath,
                           unumpy as unp)
from uncertainties.unumpy import (nominal_values as noms,
                                  std_devs as stds)
from peakdetect import gauss, get_maxima, fit_gauss
from unbinner import unbinned_array

def E(x, m, b):
    """ Define Calibration-function E(x) = m * x + b
        which takes a channel as input and should return
        the linear corresponding Energy.
    """
    return m * x + b

if __name__ == "__main__":
    # Read Eu-Data
    eu_dist = np.genfromtxt('../data/Europium.txt', unpack=True)[:5000]
    eu_time = 2964
    bkg_dist = np.genfromtxt('../data/Leermessung.txt', unpack=True)[:5000]
    bkg_time = 68580

    # find peaks
    vetos = np.array(
        [[0, 350],
         [400, 600]]
    )
    maxima = get_maxima(eu_dist, vetos=vetos)

    # get content of each maximum and plot gaussian fit
    # position, width and content will be stored into fit_results
    fit_results = np.zeros([len(maxima), 6])
    for index, (i, val) in enumerate(maxima):
        xs = np.arange(0, len(eu_dist))
        fig, ax = plt.subplots(1, 1)
        x, s, I = fit_gauss(xs, eu_dist, i, ax=ax, fit_width=15)
        fit_results[index] = [x.n, x.s, s.n, s.s, I.n, I.s]
        ax.set_xlabel('Kanal')
        ax.set_ylabel('Ereignisse')
        ax.legend(loc='best')
        fig.savefig('../build/plots/calibration-'+str(i)+'.pdf')
        fig.clf()

    # sort results and save as txt/tex
    ind = np.lexsort((fit_results[:,1], fit_results[:,0]))
    fit_results = fit_results[ind]
    np.savetxt('../build/data/calibration-maxima.txt', fit_results,
               header='mean\tmean_err\twidth\twidth_err\tI\tI_err')
    np.savetxt('../build/data/calibration-maxima.tex', fit_results, fmt='%.4f',
               delimiter=' & ', newline=' \\\\\n')

    # plot and mark maxima
    for maximum in fit_results:
        plt.axvline(maximum[0], linestyle='dashed', color='#0EB3EB', zorder=-10)
    plt.hist(unbinned_array(eu_dist), bins=200,
            histtype='stepfilled', edgecolor='none')
    plt.xlabel('Kanal')
    plt.ylabel('Ereignisse')
    plt.savefig('../build/plots/eu_dist.pdf')
    plt.clf()

    eu_energy, eu_prob = np.genfromtxt('../data/eu_spectrum_lit.txt',
                                       unpack=True)
    coeff, covar = curve_fit(E, fit_results[:,0], eu_energy)
    errs = np.sqrt(np.diag(covar))
    plt.errorbar(fit_results[:,0], eu_energy, xerr=fit_results[:,1], fmt='+',
            label='Kalibrationswerte')
    xs = np.linspace(0, 5000)
    plt.xlabel('Kanal')
    plt.ylabel('Energien / keV')
    plt.plot(xs, E(xs, *coeff), label='Linearer Fit')
    plt.xlim(0, 5000)
    plt.ylim(ymin=0)
    plt.savefig('../build/plots/calibration.pdf')

    # store calibration values
    np.savetxt('../build/data/calibration.txt', np.array([coeff, errs]).T,
            header='coeff\terr')
