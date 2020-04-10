print("load_results_multinest.py\tstart")
from common_init import *
import scipy as sp
from matplotlib import cm
import DirectDmTargets as dddm
import numpy as np
from matplotlib.colors import LogNorm
from tqdm import tqdm
import pandas as pd

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', 150)
pd.set_option('display.max_rows', 100)

import scipy
import os
import sys
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
from IPython.core.pylabtools import figsize
figsize(8, 6)
print("load_results_multinest.py\tdone packages, get results")

##
# Results
##

# loading normal results.
results = {}
all_res = dddm.context['results_dir']
res_dirs = os.listdir(all_res)
no_result = []
load_errors = []
for i, resdir in enumerate(tqdm(res_dirs)):
    try:
        result = dddm.load_multinest_samples_from_file(all_res + '/' + resdir + '/');
    except:
        e = sys.exc_info()[0]
        print(f'Error {e} in loading {resdir}')
        load_errors.append([i, all_res + '/' + resdir, e])
        continue
    if len(result.keys()):
        result['dir'] = all_res + '/' + resdir + '/'
        results[i] = result
    else:
        no_result.append(all_res + '/' + resdir + '/')

print("load_results_multinest.py\tdone, convert to results dataframe")

#TODO
def results_to_df(res):
    '''Takes res and converts it to a pd.DataFrame'''
    df = pd.DataFrame()
    items = sorted(list(res.keys()))
    df['item'] = items
    for key in tqdm(res[np.min(list(res.keys()))].keys()):
        if key in ['samples', 'weights', 'weightedsamples']:
            continue
        if key == 'config' or key == 'res_dict':
            for sub_key in res[items[0]][key].keys():
                if sub_key == 'prior':
                    for sub_sub_key in res[items[0]][key][sub_key].keys():
                        if type(res[items[0]][key][sub_key][sub_sub_key]) == dict:
                            for sub_sub_sub_key in res[items[0]][key][sub_key][sub_sub_key].keys():
                                try:
                                    df[key + '_' + sub_key + '_' + sub_sub_key + '_' + sub_sub_sub_key] = [
                                        res[it][key][sub_key][sub_sub_key][sub_sub_sub_key] for it in items]
                                except KeyError:
                                    pass
                        else:
                            df[key + '_' + sub_key + '_' + sub_sub_key] = [res[it][key][sub_key][sub_sub_key] for it in
                                                                           items]
                else:
                    try:
                        df[key + '_' + sub_key] = [res[it][key][sub_key] for it in items]
                    except KeyError:
                        pass

        else:
            try:
                df[key] = [res[it][key] for it in items]
            except KeyError:
                pass
    tols = []
    for it in items:
        try:
            tols.append(res[it]['config']['tol'])
        except KeyError:
            tols.append(None)
    df['tol'] = tols
    df['mw'] = 10 ** df['config_mw']
    df['n_fit_parameters'] = [len(pars) for pars in df['config_fit_parameters']]
    return df


df = results_to_df(results)
print("load_results_multinest.py\tdone, open with 'df'")


###
# Helper functions
###
def delete_empty(paths, delete=False, only_old=True):
    """Delete data from a given set of paths. Checks if they are empty and older than
    100 h if only_old == True"""
    for path in tqdm(paths):
        cmd = f"rm -rf {path}"
        if not delete:
            print(cmd)
        if os.path.exists(path) and len(os.listdir(path)) == 0 and delete:
            if only_old:
                t_create = datetime.datetime.fromtimestamp(os.path.getmtime(path))
                dt = datetime.datetime.now() - t_create
                if dt > datetime.timedelta(hours=100):
                    os.system(cmd)
            else:
                os.system(cmd)


def delete_with_note(df, note, delete=False):
    """
    :param df: pd.dataframe
    :param note: str, name of config_notes that are to be deleted
    :param delete: if True delete all with this note
    """
    mask = df.config_notes == note
    paths = df[mask]['dir']
    for path in paths:
        cmd = f"rm -rf {path}"
        print(cmd)
        if delete:
            os.system(cmd)


def delete_with_mask(df, mask, delete=False):
    paths = df[mask]['dir']
    for path in tqdm(paths):
        cmd = f"rm -rf {path}"
        print(cmd)
        if delete:
            os.system(cmd)


print("load_results_multinest.py\tIntroduced helperfunctions.\n\tSee delete_empty, delete_with_note and delete_with_mask")


###
# Plotting
###

def get_posterior(samples, weights):
    """

    :param samples: samples from nested sampling representing the posterior if weitgthed
    :param weights: weights of samples
    :return: weigthed samples
    """
    assert np.shape(weights) == np.shape(samples), "Samples and weights must have equal dimentions"
    # re-scale weights to have a maximum of one
    nweights = weights / np.max(weights)

    # get the probability of keeping a sample from the weights
    keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]
    # get the posterior samples
    return samples[keepidx, :]


def bin_center(xedges, yedges):
    return 0.5 * (xedges[0:-1] + xedges[1:]), 0.5 * (yedges[0:-1] + yedges[1:])


def get_hist(item, nbins=45, bin_range=None):
    if bin_range is None:
        bin_range = [results[item]['config']['prior']['log_mass']['range'],
                     results[item]['config']['prior']['log_cross_section']['range']
                     ]
    counts, xedges, yedges = np.histogram2d(*get_p_i(item), bins=nbins, range=bin_range)
    return counts, xedges, yedges


def get_hist_norm(item):
    counts, xedges, yedges = get_hist(item)
    return counts / np.sum(counts), xedges, yedges


def get_p_i(i):
    m, sig = results[i]['weighted_samples'].T[:2]
    return np.array([m, sig])


def combine_normalized(items, **plot_kwargs):
    X, Y = np.meshgrid(*get_hist_norm(items[0])[1:])
    for i in items:
        c, _, _ = get_hist_norm(i)
        plt.pcolor(X, Y, c.T, norm=LogNorm(vmin=1e-4, vmax=1), **plot_kwargs)
    plt.colorbar()


def pow10(x):
    return 10 ** x


def confidence_plot(items, text_box=False, bin_range=None, nsigma=2, nbins=50):
    print("DEPRICATED use two_confidence_plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    if bin_range == None:
        bin_range = [results[items[0]]['config']['prior']['log_mass']['range'],
                     results[items[0]]['config']['prior']['log_cross_section']['range']
                     ]

    for k, item in enumerate(items):  # , 78, 110
        x, y = get_p_i(item)
        # Make a 2d normed histogram
        H, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=bin_range, normed=True)
        # Find levels by summing histogram to objective
        norm = H.sum()  # Find the norm of the sum
        # Set contour levels
        contour3 = 0.99
        contour2 = 0.95
        contour1 = 0.68

        # Take histogram bin membership as proportional to Likelihood
        # This is true when data comes from a Markovian process
        def objective(limit, target):
            w = np.where(H > limit)
            count = H[w]
            return count.sum() - target

        target1 = norm * contour1
        level1 = scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
        levels = [level1]
        if nsigma > 1:
            target2 = norm * contour2
            level2 = scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
            levels.append(level2)
            if nsigma > 2:
                target3 = norm * contour3
                level3 = scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))
                levels.append(level3)
            if nsigma > 3:
                print('Nsigma too big')
        levels.reverse()
        levels.append(H.max())

        # Pass levels to normed kde plot
        def av_levels(x):
            return [(x[i] + x[i + 1]) / 2 for i in range(len(x) - 1)]

        if levels[0] == levels[1]:
            print("ERRRRRRRRR\n\n")
            print(levels)
            levels[0] /= 1.01
            levels = np.unique(levels)
            print(levels)
        sns_ax = sns.kdeplot(x, y, shade=True, ax=ax, n_levels=levels, cmap="viridis", normed=True, cbar=False,
                             vmin=levels[0], vmax=levels[-1])

        if k == 0:
            fit_kwargs = {'label': 'best fit'}
            bench_kwargs = {'label': 'benchmark value'}
            cbar = ax.figure.colorbar(sns_ax.collections[0])
            cbar.set_ticks(av_levels(np.linspace(0, 1, nsigma + 1)))
            col_labels = ['$3\sigma$', '$2\sigma$', '$1\sigma$'][3 - nsigma:]
            cbar.set_ticklabels(col_labels)
            cbar.set_label("Posterior probability")

    plt.scatter(np.mean(x), np.mean(y), c='black', marker='+', **fit_kwargs)
    plt.scatter(results[item]['config']['mw'], results[item]['config']['sigma'], c='blue', marker='x', **bench_kwargs)

    secax = ax.secondary_xaxis('top', functions=(pow10, np.log10))

    if 'migd' in results[items[0]]['config']['detector']:
        x_ticks = [0.01, 0.1, 1, 3, 5]
    else:
        x_ticks = [15, 25, 50, 100, 250, 500, 1000]
    for x_tick in x_ticks: ax.axvline(np.log10(x_tick), alpha=0.1)
    secax.set_ticks(x_ticks)
    plt.xlim(np.log10(x_ticks[0]), np.log10(x_ticks[-1]))
    plt.xlabel("$\log_{10}(M_{\chi}$ $[GeV/c^{2}]$)")
    secax.set_xlabel("$M_{\chi}$ $[GeV/c^{2}]$")
    plt.ylabel("$\log_{10}(\sigma_{S.I.}$ $[cm^{2}]$)")
    plt.legend(loc='upper right')

    if text_box:
        plt.text(0.05, 0.95, text_box, transform=ax.transAxes, alpha=0.5,
                 bbox=dict(facecolor="white", boxstyle="round"))


def find_largest_posterior(df, sig=-38, mw=1, fix_nlive=None):
    results = []
    for nparam in [2, 5]:
        for halo in ['shm', 'shielded_shm']:
            mask = (
                    (df['n_fit_parameters'] == nparam) &
                    (df['config_halo_model'] == halo) &
                    (df['mw'] == mw) &
                    (df['config_sigma'] == sig))
            if fix_nlive:
                mask = mask & (df['config_nlive'] == fix_nlive)
            sel_df = df[mask].sort_values('config_nlive')

            maskA = sel_df['config_halo_model'] == 'shielded_shm'
            maskB = np.array(['VerneSHM' in model for model in sel_df['config_halo_model'].values])
            mask2 = maskA | maskB
            print(np.sum(maskA), np.sum(maskB), np.sum(mask2))
            if halo == 'shielded_shm':
                sel_df = sel_df[mask2]
            else:
                sel_df = sel_df[~mask2]
            print(f'{halo} for {nparam} pars @ s = {sig}, m = {mw}')
            if len(sel_df):
                print(sel_df[['item', 'mw', 'config_sigma', 'config_nlive']][-2:-1])
                results.append(sel_df['item'].values[-1])
    return results


def get_info(i):
    info = ""
    for str_inf in ['detector', 'notes', 'start', 'fit_time', 'save_intermediate', 'earth_shielding', 'nlive', 'tol']:
        try:
            info += f"\n{str_inf} = %s" % results[i]['config'][str_inf]
            if str_inf == 'start':
                info = info[:-7]
            if str_inf == 'fit_time':
                info += 's (%.1f h)' % (results[i]['config'][str_inf] / 3600.)
        except KeyError:
            # We were trying to load something that wasn't saved in the config file, ignore it for now.
            pass
    info += '\n\n--prior--'
    for it, val in results[i]['config']['prior'].items():
        if it == 'k':
            continue
        if val['prior_type'] == 'gauss':
            info += f'\n{it} = {val["mean"]} +/- {val["std"]}'
            info += f' | range = {val["range"]}'
        else:
            info += f'\n{it}_range = {val["range"]}'
    info += '\n\n--fit--'
    for it, val in results[i]['res_dict'].items():
        info += f'\n{it} = {val}'
    if '\n' == info[:1]:
        info = info[1:]
    return info


def get_savename_title(i, save_label=''):
    this_df = df[df['item'] == i]
    print(this_df[['item', 'mw', 'config_sigma', 'n_fit_parameters', 'config_halo_model', 'config_nlive']])
    title = f'$m_w$={this_df["mw"].values[0]}' + ' $Gev/c^{2}$'
    title += f', $\log(\sigma)$ = {this_df["config_sigma"].values[0]}\n'
    title += f'model ={this_df["config_halo_model"].values[0]}'
    title += f', nfit={this_df["n_fit_parameters"].values[0]}'
    name = str(save_label)
    name += f'mw-{this_df["mw"].values[0]}_s-{this_df["config_sigma"].values[0]}_'
    name += f'{this_df["config_halo_model"].values[0]}_n-{this_df["n_fit_parameters"].values[0]}'
    name += f'_{i}'
    return name, title


def overlay_hist_confidence_info(i, save_label='', bin_range=None, save_dir='figures/', ):
    det = results[i]['config']['detector']
    name, title = get_savename_title(i, save_label)
    bin_range = [results[i]['config']['prior']['log_mass']['range'],
                 results[i]['config']['prior']['log_cross_section']['range']] if bin_range is None else bin_range
    confidence_plot([i], text_box=f'{det}-detector', nsigma=2, nbins=50, bin_range=bin_range)
    plt.title(f'{title}')
    combine_normalized([i], **{"alpha": 0.3})
    info = get_info(i)

    ax = plt.gca()
    plt.text(1.6, 1, info, transform=ax.transAxes, fontsize=12, bbox=dict(facecolor="white", boxstyle="round"),
             verticalalignment='top')
    if bin_range:
        plt.xlim(*bin_range[0])
        plt.ylim(*bin_range[1])
    if os.path.exists(save_dir):
        plt.savefig(f"{save_dir}/{name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_dir}/{name}.pdf", dpi=300, bbox_inches="tight")
        dddm.multinest_corner(results[i], save_dir + name)
    else:
        print(f"Warning! No {save_dir}")
    plt.show()

def get_binrange(i):
    return results[i]['config']['prior']['log_mass']['range'], results[i]['config']['prior']['log_cross_section'][
        'range']

def one_confidence_plot(i, save_label='', save_dir='figures/', corner = False):
    det = results[i]['config']['detector']
    name, title = get_savename_title(i, save_label)
    _one_confidence_plot(i, nbins=200, text_box=str(det), smoothing=True)
    plt.title(f'{title}')
    info = get_info(i)
    ax = plt.gca()
    plt.text(1.3, 1, info, transform=ax.transAxes, fontsize=12, bbox=dict(facecolor="white", boxstyle="round"),
             verticalalignment='top')
    if os.path.exists(save_dir):
        plt.savefig(f"{save_dir}/{name}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{save_dir}/{name}.pdf", dpi=300, bbox_inches="tight")
        if corner:
            dddm.multinest_corner(results[i], save_dir + name)
    else:
        print(f"Warning! No {save_dir}")
    # plt.show()


def _one_confidence_plot(item, text_box=False,
                        bin_range=None, nsigma=3,
                        smoothing=None,
                        nbins=50):
    if bin_range == None:
        bin_range = [results[item]['config']['prior']['log_mass']['range'],
                     results[item]['config']['prior']['log_cross_section']['range']
                     ]
    x, y = get_p_i(item)
    # Make a 2d normed histogram
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins, range=bin_range, normed=True)
    if smoothing:
        H = sp.ndimage.filters.gaussian_filter(
            H.T,
            [np.sqrt(nbins) / 10, np.sqrt(nbins) / 10],
            mode='constant')
    else:
        H = H.T

    H = H / np.sum(H)
    X, Y = bin_center(xedges, yedges)
    _confidence_plot(item, X, Y, H, bin_range, text_box=text_box, nsigma=nsigma)

def two_confidence_plot(items, text_box=False,
                        bin_range=None, nsigma=3,
                        smoothing=None,
                        nbins=50):
    if bin_range == None:
        bin_range = [results[items[0]]['config']['prior']['log_mass']['range'],
                     results[items[0]]['config']['prior']['log_cross_section']['range']
                     ]
    hists = {}
    for k, item in enumerate(items):  # , 78, 110
        x, y = get_p_i(item)
        # Make a 2d normed histogram
        hists[k], xedges, yedges = np.histogram2d(x, y, bins=nbins, range=bin_range, normed=True)
        if smoothing:
            hists[k] = sp.ndimage.filters.gaussian_filter(
                hists[k].T,
                [np.sqrt(nbins) / 10, np.sqrt(nbins) / 10],
                mode='constant')
        else:
            hists[k] = hists[k].T
    if len(items) == 2:
        H = hists[0] * hists[1]
    else:
        raise ValueError(f'Len items is {len(items)}')
    H = H / np.sum(H)
    X, Y = bin_center(xedges, yedges)
    _confidence_plot(item, X, Y, H, bin_range, text_box=text_box, nsigma=nsigma)


    
def _confidence_plot(item, X, Y, H, bin_range, text_box=False, nsigma=3, cmap = cm.inferno_r):
    # cmap = cm.viridis
    # cmap = cm.gnuplot2_r
    
    xmean, xerr = weighted_avg_and_std(X, np.mean(H, axis=0))
    ymean, yerr = weighted_avg_and_std(Y, np.mean(H, axis=1))
    print(f'X mean, std {xmean, xerr}')
    print(f'Y mean, std {ymean, yerr}')
    norm = H.sum()  # Find the norm of the sum
    # Set contour levels
    contour3 = 0.99
    contour2 = 0.95
    contour1 = 0.68

    # Take histogram bin membership as proportional to Likelihood
    # This is true when data comes from a Markovian process
    def objective(limit, target):
        w = np.where(H > limit)
        count = H[w]
        return count.sum() - target

    target1 = norm * contour1
    level1 = scipy.optimize.bisect(objective, H.min(), H.max(), args=(target1,))
    levels = [level1]
    # Find levels by summing histogram to objective
    if nsigma > 1:
        target2 = norm * contour2
        level2 = scipy.optimize.bisect(objective, H.min(), H.max(), args=(target2,))
        levels.append(level2)
        if nsigma > 2:
            target3 = norm * contour3
            level3 = scipy.optimize.bisect(objective, H.min(), H.max(), args=(target3,))
            levels.append(level3)
        if nsigma > 3:
            print('Nsigma too big')

    levels.reverse()
    levels.append(H.max())

    if levels[0] == levels[1]:
        print("ERRRRRRRRR\n\n")
        print(levels)
        levels[0] /= 1.01
        levels = np.unique(levels)
        print(levels)

    contours = plt.contourf(X, Y, H, (0,),
                            levels=levels,
                            cmap=cm.get_cmap(cmap, len(levels) - 1), norm = LogNorm()
                            )
    ax = plt.gca()
    cset2 = ax.contour(X, Y, H, contours.levels, colors='k')

    # We don't really need dashed contour lines to indicate negative
    # regions, so let's turn them off.
    for c in cset2.collections:
        c.set_linestyle('solid')

    plt.errorbar(xmean, ymean, xerr, yerr,
                 c='red', capsize=3, label = 'best fit', marker='o')

    plt.scatter(results[item]['config']['mw'], results[item]['config']['sigma'],
                c='blue', marker='x', label = 'benchmark value')

    cbar = ax.figure.colorbar(contours)
    col_labels = ['$3\sigma$', '$2\sigma$', '$1\sigma$'][3 - nsigma:]
    cbar.set_ticklabels(col_labels)
    cbar.set_label("Posterior probability")

    secax = ax.secondary_xaxis('top', functions=(pow10, np.log10))

    if 'migd' in results[item]['config']['detector']:
        x_ticks = [0.01, 0.1, 1, 3, 5, 10, 50, 100]
    else:
        x_ticks = [15, 25, 50, 100, 250, 500, 1000]
    for x_tick in x_ticks:
        ax.axvline(np.log10(x_tick), alpha=0.1)
    secax.set_ticks(x_ticks, rotation = 90)
    plt.xlim(*bin_range[0])
    plt.ylim(*bin_range[1])
    plt.xlabel("$\log_{10}(M_{\chi}$ $[GeV/c^{2}]$)")
    secax.set_xlabel("$M_{\chi}$ $[GeV/c^{2}]$")
    plt.ylabel("$\log_{10}(\sigma_{S.I.}$ $[cm^{2}]$)")
    plt.legend(loc='upper right')

    if text_box:
        plt.text(0.05, 0.95, text_box, transform=ax.transAxes, alpha=0.5,
                 bbox=dict(facecolor="white", boxstyle="round"))
    return  xmean, xerr, ymean, yerr

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return (average, np.sqrt(variance))


def match_other_item(i, verbose=False, diff_det_type=False):
    this_df = df[df.item == i]
    if len(this_df) == 0:
        print(f"WARNING: NO item {i}")
        return
    assert len(this_df) < 2, f'found >1 entries in df for {i}'
    sub_df = df.copy()
    for key in ['config_poisson', 'config_n_energy_bins', 'config_earth_shielding', 'config_save_intermediate',
                'config_prior_log_mass_range', 'config_prior_log_mass_prior_type', 'config_prior_log_mass_param',
                'config_prior_log_cross_section_range', 'config_prior_log_cross_section_prior_type',
                'config_prior_log_cross_section_param', 'config_prior_density_range',
                'config_prior_density_prior_type', 'config_prior_density_mean', 'config_prior_density_std',
                'config_prior_density_param', 'config_prior_density_dist', 'config_prior_v_0_range',
                'config_prior_v_0_prior_type', 'config_prior_v_0_mean', 'config_prior_v_0_std',
                'config_prior_v_0_param', 'config_prior_v_0_dist', 'config_prior_v_esc_range',
                'config_prior_v_esc_prior_type', 'config_prior_v_esc_mean', 'config_prior_v_esc_std',
                'config_prior_v_esc_param', 'config_prior_v_esc_dist', 'config_prior_k_range',
                'config_prior_k_prior_type', 'config_prior_k_param', 'config_prior_k_dist',
                'config_v_0', 'config_v_esc', 'config_density', 'config_mw', 'config_sigma',
                'config_halo_model', 'config_spectrum_class',
                'config_nlive', 'n_fit_parameters']:
        val = this_df[key].values[0]
        if 'dist' in key:
            continue
        if diff_det_type and 'prior_log' in key:
            continue
        if verbose: print(f'looking for {key}:{val}\t\ttotlen:{len(sub_df)}')
        if np.iterable(val) and len(val) == 2:
            mask = [((_val[0] == val[0]) and (_val[1] == val[1])) for _val in sub_df[key]]
        else:
            mask = sub_df[key] == val
        sub_df = sub_df[mask]
        if len(sub_df) == 0:
            break

    sub_df = sub_df[~(sub_df['config_detector'] == this_df['config_detector'].values[0])]
    if len(sub_df) == 0:
        print("WARNING: NO MATCH")
        return
    for _det in ['Ge', 'Xe']:
        if _det in this_df['config_detector'].values[0]:
            mask = np.array([_det in det for det in sub_df['config_detector']])
            if not diff_det_type:
                sub_df = sub_df[mask]
            else:
                try:
                    sub_df = sub_df[~mask]
                except TypeError:
                    print(sub_df, mask)
                    return sub_df, mask
    if len(sub_df) == 1:
        return sub_df['item'].values[0]
    elif len(sub_df) == 0:
        print("WARNING: NO MATCH")
        return
    else:
        print("WARNING: MULTIPLE MATHCES")
        return sub_df