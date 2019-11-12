# Some basic functions for plotting et cetera

import numpy as np
import emcee
import pandas as pd
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt

from . import __init__
from .halo import *
from .statistics import *
from .detector import *


def error_bar_hist(ax, data, data_range=None, nbins=50, **kwargs):
    x, y, yerr = hist_data(data, data_range, nbins)
    ax.errorbar(x, y, yerr=yerr, capsize=3, marker='o', **kwargs)


def hist_data(data, data_range=None, nbins=50):
    if data_range is not None:
        data_range = [np.min(data), np.max(data)]
    else:
        assert_str = "make sure data_range is of fmt [x_min, x_max]"
        assert (type(data_range) == list or type(data_range) == tuple) and len(
            data_range) == 2, assert_str

    counts, bin_edges = np.histogram(data, range=data_range, bins=nbins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    x, y, yerr = bin_centers, counts, np.sqrt(counts)
    return x, y, yerr


def simple_hist(y):
    plt.figure(figsize=(19, 6))
    ax = plt.gca()
    data_all = hist_data(y)
    ax.plot(data_all[0], data_all[1], linestyle='steps-mid',
            label="Pass through")
    error_bar_hist(ax, y)


def ll_element_wise(x, y, clip_val=-1e4):
    rows = len(x)
    cols = len(x[0])
    r = np.zeros((rows, cols))
    for i in tqdm(range(rows)):
        for j in range(cols):
            r[i][j] = log_likelihood_function(x[i][j], y[i][j])
    return np.clip(r, clip_val, 0)


def show_ll_function(npoints = 1e4, clip_val = -1e4, min_val = 0.1):
    from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, \
        title, show
    from matplotlib.colors import LogNorm
    x = np.arange(0, npoints, 1)
    y = np.arange(0, npoints, 1)
    X, Y = meshgrid(x, y)  # grid of point
    Z = -ll_element_wise(X, Y, clip_val)
    im = imshow(Z, cmap=cm.RdBu,
                norm=LogNorm(min_val, -clip_val))  # drawing the function
    colorbar(im, label='$-\mathcal{L}$')  # adding the colobar on the right
    title('$-\mathcal{L}$ clipped at %i' % (clip_val))
    plt.xlabel("Nb")
    plt.ylabel("Nr")
    show()


def plt_ll_sigma_spec(det='Xe'):
    use_SHM = SHM()
    events = GenSpectrum(50, 1e-45, use_SHM, detectors[det])
    data = events.get_data(poisson=False)
    sigma_nucleon = np.linspace(0.11 * 1e-45, 10 * 1e-45, 30)
    model = lambda x: GenSpectrum(50, x, use_SHM, detectors[det]).get_data(
        poisson=False)
    plr = [log_likelihood(model(x), data['counts']) for x in
           tqdm(sigma_nucleon)]
    plt.plot(sigma_nucleon, plr, linestyle = 'steps-mid')
    plt.axvline(1e-45)
    plt.xlim(sigma_nucleon[0], sigma_nucleon[-1])
    plt.ylim(np.min(plr), np.max(plr))


def plt_ll_mass_spec(det='Xe'):
    use_SHM = SHM()
    mass = np.linspace(5, 100, 100)
    events = GenSpectrum(50, 1e-45, use_SHM, detectors[det])
    # TODO should add the poissonian noise
    xe_data = events.get_data(poisson=False)
    model = lambda x: GenSpectrum(x, 1e-45, use_SHM, detectors[det]).get_data(
        poisson=False)
    plr = [log_likelihood(model(x), xe_data['counts']) for x in tqdm(mass)]

    mass, plr = remove_nan(mass, plr), remove_nan(plr, mass)
    assert len(mass) > 0, "empty data remains"

    plt.plot(mass, plr, linestyle = 'steps-mid')
    plt.axvline(50)
    # plt.xlim(sigma_nucleon[0], sigma_nucleon[-1])

    plt.ylim(np.min(plr), np.max(plr))


def plt_ll_sigma_det(det='Xe'):
    use_SHM = SHM()
    events = DetectorSpectrum(50, 1e-45, use_SHM, detectors[det])
    data = events.get_data(poisson=False)
    sigma_nucleon = np.linspace(0.11 * 1e-45, 10 * 1e-45, 30)
    model = lambda x: DetectorSpectrum(50, x, use_SHM, detectors[det]).get_data(
        poisson=False)
    plr = [log_likelihood(model(x), data['counts']) for x in
           tqdm(sigma_nucleon)]
    plt.plot(sigma_nucleon, plr, linestyle = 'steps-mid')
    plt.axvline(1e-45)
    plt.xlim(sigma_nucleon[0], sigma_nucleon[-1])
    plt.ylim(np.min(plr), np.max(plr))


def plt_ll_mass_det(det='Xe'):
    use_SHM = SHM()
    mass = np.linspace(5, 100, 100)
    events = DetectorSpectrum(50, 1e-45, use_SHM, detectors[det])
    # TODO should add the poissonian noise
    data = events.get_data(poisson=False)
    model = lambda x: DetectorSpectrum(
        x, 1e-45, use_SHM, detectors[det]).get_data(poisson=False)
    plr = [log_likelihood(model(x), data['counts']) for x in tqdm(mass)]
    mass, plr = remove_nan(mass, plr), remove_nan(plr, mass)
    assert len(mass) > 0, "empty data remains"
    plt.plot(mass, plr, linestyle = 'steps-mid')
    plt.axvline(50)
    plt.ylim(np.min(plr), np.max(plr))


def plt_priors(itot=100):
    priors = get_priors()
    for key in priors.keys():
        par = priors[key]['param']
        dist = priors[key]['dist']
        data = [dist(par) for i in range(itot)]
        if priors[key]['prior_type'] == 'gauss':
            plt.axvline(priors[key]['mean'], c='r')
            plt.axvline(priors[key]['mean'] - priors[key]['std'], c='b')
            plt.axvline(priors[key]['mean'] + priors[key]['std'], c='b')
        plt.hist(data, bins=100)
        plt.title(key)
        plt.show()


# No detector resolution
def plot_spectrum(data, color = 'blue', label = 'label', linestyle = 'none', plot_error = True):
    plt.errorbar(data['bin_centers'], data['counts'],
                xerr=(data['bin_left'] - data['bin_right'])/2,
                yerr = np.sqrt(data['counts']) if plot_error else np.zeros(len(data['counts'])),
                color = color,
                linestyle = linestyle,
                capsize = 2,
                marker = 'o',
                label = label,
                markersize=2
                )
