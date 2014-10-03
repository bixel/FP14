import numpy as np
import matplotlib.pyplot as plt
from peakdetect import get_maxima, fit_gauss
from unbinner import unbinned_array
from calibration import E

if __name__ == '__main__':
    cal_coeff, cal_errs = np.genfromtxt('../build/data/calibration.txt',
            unpack=True)
    cs_dist = np.genfromtxt('../data/Caesium.txt', unpack=True)[:5000]
    xs = E(np.arange(0, 5000), *cal_coeff)
    print(len(cs_dist), len(xs))
    maxima = get_maxima(cs_dist, n_max=1)
    print(maxima)
    fig, ax = plt.subplots(1, 1)
    mu, sigma, A = fit_gauss(xs, cs_dist, xs[maxima[0, 0]], ax=ax, fit_width=5)
    ax.set_xlabel('Energie / keV')
    ax.set_ylabel('Ereignisse')
    ax.legend(loc='best')
    fig.savefig('../build/plots/caesium_gauss.pdf')
    
