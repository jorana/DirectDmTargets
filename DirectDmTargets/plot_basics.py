# Some basic functions for plotting et cetera

import matplotlib.pyplot as plt
import numpy as np
from .statistics import *
from tqdm import tqdm


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


def show_ll_function():
    from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, axis, \
        title, show
    from matplotlib.colors import LogNorm
    x = np.arange(0, 1e4, 1)
    y = np.arange(0, 1e4, 1)
    X, Y = meshgrid(x, y)  # grid of point
    clip_val = -1e4
    Z = -ll_element_wise(X, Y, clip_val)
    im = imshow(Z, cmap=cm.RdBu,
                norm=LogNorm(0.1, -clip_val))  # drawing the function
    colorbar(im, label='$-\mathcal{L}$')  # adding the colobar on the right
    title('$-\mathcal{L}$ clipped at %i' % (clip_val))
    plt.xlabel("Nb")
    plt.ylabel("Nr")
    show()
