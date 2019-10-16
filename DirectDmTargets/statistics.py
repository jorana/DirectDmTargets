import numpy as np
# import emcee
import pandas as pd
# import scipy
from .halo import *
from .detector import *

# TODO
use_SHM = SHM()

priors = {
    'log_mass': {'range': [0.1, 3], 'prior_type': 'flat'},
    'log_cross_secion': {'range': [-10, -6], 'prior_type': 'flat'},
    'density': {'range': [0.001, 0.9], 'prior_type': 'gauss', 'mean': 0.4, 'std': 0.1},
    'v_0': {'range': [80, 380], 'prior_type': 'gauss', 'mean': 230, 'std': 30},
    'v_esc': {'range': [379, 709], 'prior_type': 'gauss', 'mean': 544, 'std': 33},
    'k': {'range': [0.5, 3.5], 'prior_type': 'flat'}
}

for key in priors.keys():
    param = priors[key]
    if param['prior_type'] == 'flat':
        param['param'] = param['range']
        param['dist'] = lambda x: flat_prior(x)
    elif param['prior_type'] == 'gauss':
        param['param'] = param['mean'], param['std']
        param['dist'] = lambda x: gaus_prior(x)


def approx_log_fact(n):
    """take the approximate logarithm of factorial n for large n

    :param n: the number n
     :return:  ln(n!)"""
    assert n >= 0, f"Only take the logarithm of n>0. (n={n})"

    # if n is small, there is no need for approximation
    if n < 100:
        # gamma equals factorial for x -> x +1 and also returns results for non-integers
        return np.log(np.math.gamma(n + 1))

    # Stirling's approximation <https://nl.wikipedia.org/wiki/Formule_van_Stirling>
    return n * np.log(n) - n


def log_likelihood_function(Nb, Nr):
    """return the ln(likelihood) for Nb expected events and Nr observed events

    :param Nb: expected events
    :param Nr: observed events
    :return: ln(likelihood)
    """
    # https://www.wolframalpha.com/input/?i=simplify+ln%28R%5Eb+%2F+b%21+exp%28-b%29%29
    return Nb * np.log((Nr)) - approx_log_fact(Nb) - Nr


def log_likelihood(model, data):
    """
    :param model: pandas dataframe containing the number of counts in bin i
    :param data: pandas dataframe containing the number of counts in bin i
    :return: product of the likelihoods of the bins
    """

    assert data.shape == model.shape, f"Data and model should be of same dimensions (now {data.shape, model.shape})"
    assert_str = f"please insert pd.dataframe for data ({type(data)}) and model ({type(model)})"
    assert type(data) == type(model) == pd.DataFrame, assert_str

    res = 1
    for i in range(data.shape[0]):
        Nr = data['counts'][i]
        Nb = model['counts'][i]
        res_bin = log_likelihood_function(Nb, Nr)

        # TODO check this lower bound suitable?
        res += np.maximum(res_bin, -10e9)
    if res < -10e9:
        print('log_likelihood::\tbadly defined likelihood. Something might be wrong.')
    return res


def not_nan_inf(x):
    """
    :param x: float or array
    :return: array of True and/or False indicating if x is nan/inf
    """

    if np.shape(x) == () and x == None:
        x = np.nan
    try:
        return np.isnan(x) ^ np.isinf(x)
    except TypeError:
        return np.array([not_nan_inf(xi) for xi in x])


def masking(x, mask):
    """
    :param x: float or array
    :param mask: array of True and/or False
    :return: x[mask]
    """
    assert len(x) == len(mask), f"match length mask {len(mask)} to length array {len(x)}"
    try:
        return x[mask]
    except TypeError:
        return np.array([x[i] for i in range(len(x)) if mask[i]])


def remove_nan(x, maskable=False):
    """
    :param x: float or array
    :param maskable: array to take into consideration when removing NaN and/or inf from x
    :return: x where x is well defined (not NaN or inf)
    """
    if maskable != False:
        assert len(x) == len(maskable), f"match length maskable {len(maskable)} to length array {len(x)}"
    if type(maskable) == bool and maskable == False:
        mask = ~not_nan_inf(x)
        return masking(x, mask)
    else:
        return masking(x, ~not_nan_inf(maskable) ^ not_nan_inf(x))


def flat_prior(_range):
    return np.random.uniform(_range[0], _range[1])


def gaus_prior(_param):
    mu, sigma = _param
    return np.random.normal(mu, sigma)


def log_probability(theta, x, y, x_name):
    lp = log_prior(theta, x_name)
    if not np.isfinite(lp):
        return -np.inf
    model = eval_log_likelihood(theta)

    ll = log_likelihood(model, x, y)
    if np.isnan(lp + ll):
        raise ValueError(f"Returned NaN from likelihood. lp = {lp}, ll = {ll}")
    return lp + ll


def log_flat(x, x_name):
    a, b = priors[x_name]['param']
    try:
        if a < x < b:
            return 0
        else:
            return -np.inf
    except ValueError:
        result = np.zeros(len(x))
        mask = (x > a) & (x < b)
        result[~mask] = -np.inf
        return result


def log_gauss(x, x_name):
    mu, sigma = priors[x_name]['param']
    return -0.5 * np.sum((x - mu) ** 2 / (sigma ** 2) + np.log(sigma ** 2))


def log_prior(x, x_name):
    if priors[x_name]['prior_type'] == 'flat':
        if 'log' in x_name:
            return log_flat(np.log10(x), x_name)
        else:
            return log_flat(x, x_name)
    elif priors[x_name]['prior_type'] == 'gauss':
        return log_gauss(x, x_name)
    else:
        raise TypeError(f"unknown prior type '{priors[x_name]['prior_type']}', choose either gauss or flat")


def eval_log_likelihood(theta):
    x = theta
    if np.shape(theta) == (1,):
        x = theta[0]
    return GenSpectrum(x, 10e-45, use_SHM, detectors['Xe']).get_data(poisson=False)


def log_likelihood(model, x, y):
    res = 1
    for i in range(len(x)):

        Nr = y[i]
        Nb = model['counts'][i]
        # TO DO round Nb to int is okay?
        ## https://www.wolframalpha.com/input/?i=simplify+ln%28R%5Eb+%2F+b%21+exp%28-b%29%29

        res_bin = Nb * np.log((Nr)) - approx_log_fact(Nb) - Nr
        if np.isnan(res_bin):
            raise ValueError(f"Returned NaN in bin {i}. Below follows data dump.\n"
                             f"i = {i}, Nb, Nr = {Nb, Nr}\n"
                             f"log(Nr) = {np.log((Nr))}, Nb! = {approx_log_fact(Nb)}")
        # TODO check this lowerbound
        res += np.maximum(res_bin, -10e9)
    return res
