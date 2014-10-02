import numpy as np
from scipy.optimize import curve_fit
from uncertainties import ufloat

def gauss(x, A, mu, sigma, B):
    """ define gaussian with mean mu and width sigma
    """
    return A * np.exp(-(x-mu)**2 / (2 * sigma**2)) \
            / (np.sqrt(2 * np.pi) * sigma) \
            + B


def fit_gauss(x, y, pos_x, fit_width=30, yerr=0, ax=None):
    """ fit gaussian function to given x- and y-data at position pos_x

        @return mean, width and content of gauss
    """
    import matplotlib.pyplot as plt
    import numpy.ma as ma
    mask = ma.masked_inside(x, pos_x - fit_width, pos_x + fit_width).mask
    x = x[mask]
    y = y[mask]
    p0=[np.median(x), pos_x, 1, 5]
    coeff, covar = curve_fit(gauss, x, y, p0=p0)
    errs = np.sqrt(np.diag(covar))
    mu = ufloat(coeff[1], errs[1])
    A = ufloat(coeff[0], errs[0])
    sigma = ufloat(coeff[2], errs[2])
    if ax is not None:
        xs = np.linspace(min(x), max(x), 200)
        ax.plot(xs, gauss(xs, *coeff), label='Gaußscher Fit')
        ax.errorbar(x, y, yerr=yerr, fmt='+', label='Datenpunkte')
    return mu, sigma, A


def get_maxima(dist, n_max=10, vetos=[]):
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
        maxima[index] = max_val_index, dist[max_val_index]
        no_max_dist[max_val_index - 10:max_val_index + 10] = 0
    return maxima

