"""Some basic functions for plotting et cetera. Used to for instance check that
the likelihood function is well behaved"""

import colorsys
import os

import matplotlib.pyplot as plt
import numpy as np
from DirectDmTargets import statistics, halo, detector, utils
from tqdm import tqdm


def error_bar_hist(ax, data, data_range=None, nbins=50, **kwargs):
    x, y, yerr = hist_data(data, data_range, nbins)
    ax.errorbar(x, y, yerr=yerr, capsize=3, marker='o', **kwargs)


def hist_data(data, data_range=None, nbins=50):
    if data_range is None:
        data_range = [np.min(data), np.max(data)]
    else:
        assert_str = "make sure data_range is of fmt [x_min, x_max]"
        assert isinstance(data_range, (list, tuple)) and len(
            data_range) == 2, assert_str

    counts, _bin_edges = np.histogram(data, range=data_range, bins=nbins)
    bin_centers = (_bin_edges[:-1] + _bin_edges[1:]) / 2
    x, y, yerr = bin_centers, counts, np.sqrt(counts)
    return x, y, yerr


def simple_hist(y: np.ndarray):
    plt.figure(figsize=(19, 6))
    ax = plt.gca()
    data_all = hist_data(y)
    ax.plot(data_all[0], data_all[1], drawstyle='steps-mid',
            label="Pass through")
    error_bar_hist(ax, y)


def ll_element_wise(x, y, clip_val=-1e4):
    rows = len(x)
    cols = len(x[0])
    r = np.zeros((rows, cols))
    for i in tqdm(range(rows)):
        for j in range(cols):
            r[i][j] = statistics.log_likelihood_function(x[i][j], y[i][j])
    return np.clip(r, clip_val, 0)


def show_ll_function(npoints=1e4, clip_val=-1e4, min_val=0.1):
    from pylab import meshgrid, cm, imshow, colorbar, title
    from matplotlib.colors import LogNorm
    x = np.arange(0, npoints, 1)
    y = np.arange(0, npoints, 1)
    X, Y = meshgrid(x, y)  # grid of point
    Z = -ll_element_wise(X, Y, clip_val)
    im = imshow(Z, cmap=cm.RdBu,
                norm=LogNorm(min_val, -clip_val))  # drawing the function
    colorbar(im, label=r'$-\mathcal{L}$')  # adding the colorbar on the right
    title(r'$-\mathcal{L}$ clipped at %i' % clip_val)
    plt.xlabel("Nb")
    plt.ylabel("Nr")


def plt_ll_sigma_mass(spec_clas, vary, det='Xe', bins=10, m=50, sig=1e-45):
    assert vary in ['mass', 'sig'], "use sig or mass"
    use_SHM = halo.SHM()
    events = spec_clas(m, sig, use_SHM, detector.experiment[det])
    events.n_bins = bins
    data = events.get_data(poisson=False)
    if vary == 'sig':
        plt.xlabel(r'$\sigma$ $[cm^2]$')
        plt.axvline(sig, alpha=0.5, color='red', label='truth')
        var = np.linspace(0.1 * 1e-45, 10 * 1e-45, 30)

        def model(x):
            res = spec_clas(m, x, use_SHM, detector.experiment[det])
            res.n_bins = bins
            return res.get_data(poisson=False)['counts']

    elif vary == 'mass':
        plt.xlabel('mass [GeV/$c^2$]')
        plt.axvline(m, alpha=0.5, color='red', label='truth')
        plt.axvline(33, alpha=0.1, color='black', label='binning boundary')
        var = np.concatenate((np.linspace(1, 33, 50),
                              np.linspace(33, 300, 50)))

        def model(x):
            res = spec_clas(x, sig, use_SHM, detector.experiment[det])
            res.n_bins = bins
            return res.get_data(poisson=False)['counts']
    else:
        raise ValueError(f'Can not vary {vary}')
    plr = [statistics.log_likelihood(data['counts'], model(x)) for x in
           tqdm(var)]

    plt.xlim(var[0], var[-1])
    var, plr = utils.remove_nan(var, plr), utils.remove_nan(plr, var)
    plt.plot(var, plr, drawstyle='steps-mid')
    plt.ylim(np.min(plr), np.max(plr))


def plt_ll_sigma_spec(det='Xe', bins=10, m=50, sig=1e-45):
    plt_ll_sigma_mass(
        halo.GenSpectrum,
        'sig',
        det=det,
        bins=bins,
        m=m,
        sig=sig)


def plt_ll_mass_spec(det='Xe', bins=10, m=50, sig=1e-45):
    plt_ll_sigma_mass(
        halo.GenSpectrum,
        'mass',
        det=det,
        bins=bins,
        m=m,
        sig=sig)


def plt_ll_sigma_det(det='Xe', bins=10, m=50, sig=1e-45):
    plt_ll_sigma_mass(
        detector.DetectorSpectrum,
        'sig',
        det=det,
        bins=bins,
        m=m,
        sig=sig)


def plt_ll_mass_det(det='Xe', bins=10, m=50, sig=1e-45):
    plt_ll_sigma_mass(
        detector.DetectorSpectrum,
        'mass',
        det=det,
        bins=bins,
        m=m,
        sig=sig)


def plt_priors(itot=100):
    priors = statistics.get_priors()
    for key in priors.keys():
        par = priors[key]['param']
        dist = priors[key]['dist']
        data = [dist(par) for _ in range(itot)]
        if priors[key]['prior_type'] == 'gauss':
            plt.axvline(priors[key]['mean'], c='r')
            plt.axvline(priors[key]['mean'] - priors[key]['std'], c='b')
            plt.axvline(priors[key]['mean'] + priors[key]['std'], c='b')
        plt.hist(data, bins=100)
        plt.title(key)


def plot_spectrum(data, color='blue', label='label', linestyle='none',
                  plot_error=True):
    plt.errorbar(data['bin_centers'], data['counts'],
                 xerr=(data['bin_left'] - data['bin_right']) / 2,
                 yerr=np.sqrt(data['counts']) if plot_error else np.zeros(
                     len(data['counts'])),
                 color=color,
                 linestyle=linestyle,
                 capsize=2,
                 marker='o',
                 label=label,
                 markersize=2
                 )


def get_color_from_range(val, _range=(0, 1), it=0):
    if not np.iterable(_range):
        _range = [0, _range]
    red_to_green = (val - _range[0]) / np.diff(_range)
    assert 0 <= red_to_green <= 1, f'{val} vs {_range} does not work'
    assert it <= 2
    # in HSV, red is 0 deg and green is 120 deg (out of 360);
    # divide red_to_green with 3 to map [0, 1] to [0, 1./3.]
    hue = red_to_green / 3.0
    hue += it / 3
    res = colorsys.hsv_to_rgb(hue, 1, 1)
    # return [int(255 * float(r)) for r in res]
    return [float(r) for r in res]


def save_canvas(name, save_dir='./figures', tight_layout=False):
    utils.check_folder_for_file(save_dir + '/.')
    utils.check_folder_for_file(save_dir + '/pdf/.')
    if tight_layout:
        plt.tight_layout()
    if os.path.exists(save_dir) and os.path.exists(save_dir + '/pdf'):
        plt.savefig(f"{save_dir}/{name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_dir}/pdf/{name}.pdf", dpi=300, bbox_inches="tight")
    else:
        raise FileExistsError(
            f'{save_dir} does not exist or does not have /pdf')
