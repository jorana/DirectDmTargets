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
    'log_cross_section': {'range': [-46, -42], 'prior_type': 'flat'},
    # TODO
    # 'log_cross_section': {'range': [-10, -6], 'prior_type': 'flat'},
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
    if n < 10:
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
    if Nr < 5 and Nb < 5:
        return np.log(((Nr**Nb) /np.math.gamma(Nb+1)) * np.exp(-Nr))
    # # https://www.wolframalpha.com/input/?i=simplify+ln%28R%5Eb+%2F+b%21+exp%28-R%29%29
    # else:
    return Nb * np.log(Nr) - approx_log_fact(Nb) - Nr


def log_likelihood_df(model, data):
    """
    :param model: pandas dataframe containing the number of counts in bin i
    :param data: pandas dataframe containing the number of counts in bin i
    :return: product of the likelihoods of the bins
    """

    assert data.shape == model.shape, f"Data and model should be of same dimensions (now {data.shape, model.shape})"
    assert_str = f"please insert pd.dataframe for data ({type(data)}) and model ({type(model)})"
    assert type(data) == type(model) == pd.DataFrame, assert_str

    return log_likelihood(model, data['bin_centers'], data['counts'])

def log_likelihood(model, x, y):
    """
    :param model: pandas dataframe containing the number of counts in bin i
    :param x: energy of bin i
    :param y: the number of counts in bin i
    :return: product of the likelihoods of the bins
    """

    assert len(x) == len(y) == model.shape[0], f"""Data and model should be of same dimensions (now {len(x), len(y), 
            model.shape[0]})"""
    assert_str = f"please insert pd.dataframe for model ({type(model)})"
    assert type(model) == pd.DataFrame, assert_str
    # TODO Also add the assertion error for x and y
    res = 1
    for i in range(len(x)):

        Nr = y[i]
        Nb = model['counts'][i]
        # TODO round Nb to int is okay?
        ## https://www.wolframalpha.com/input/?i=simplify+ln%28R%5Eb+%2F+b%21+exp%28-b%29%29

        res_bin = log_likelihood_function(Nb, Nr)
        if np.isnan(res_bin):
            raise ValueError(
                f"Returned NaN in bin {i}. Below follows data dump.\n"
                f"i = {i}, Nb, Nr = {Nb, Nr}\n"
                f"res_bin {res_bin}\n"
                f"log(Nr) = {np.log((Nr))}, Nb! = {approx_log_fact(Nb)}\n"
                f"log_likelihood: {log_likelihood_function(Nb, Nr)}\n"
                )
        if not np.isfinite(res_bin):
            return -np.inf
        res += res_bin
        # TODO check this lowerbound
        # res += np.maximum(res_bin, -10e9)
    # if res < -10e9:
    #     print('log_likelihood::\tbadly defined likelihood. Something might be wrong.')
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
    if type(maskable) is not bool:
        assert len(x) == len(maskable), f"match length maskable {len(maskable)} to length array {len(x)}"
    if type(maskable) is bool and maskable is False:
        mask = ~not_nan_inf(x)
        return masking(x, mask)
    else:
        return masking(x, ~not_nan_inf(maskable) ^ not_nan_inf(x))


def flat_prior(_range):
    return np.random.uniform(_range[0], _range[1])


def gaus_prior(_param):
    mu, sigma = _param
    return np.random.normal(mu, sigma)

# TODO add to class
def log_probability(theta, x, y, x_names):
    # single parameter to fit
    if type(x_names) == str:
        x_val = theta
        lp = log_prior(x_val, x_names)

    elif len(x_names) > 1:
        assert len(theta) == len(x_names), f"provide enough names ({x_names}) for the parameters (len{len(theta)})"
        lp = np.sum([log_prior(*_x) for _x in zip(theta, x_names)])
    else:
        raise TypeError(f"""incorrect format provided. Theta should be array-like for single value of x_names or
            Theta should be matrix-like for array-like x_names. Theta, x_names (provided) = {theta, x_names}""")
    if not np.isfinite(lp):
        return -np.inf
    if type(x_names) == str and x_names == 'log_mass':
        model = eval_log_likelihood_mass(theta)
    else:
        model = eval_log_likelihood(theta, x_names)

    ll = log_likelihood(model, x, y)
    if np.isnan(lp + ll):
        raise ValueError(f"Returned NaN from likelihood. lp = {lp}, ll = {ll}")
    return lp + ll

def log_probability_detector(theta, x, y, x_names):
    # single parameter to fit
    if type(x_names) == str:
        x_val = theta
        lp = log_prior(x_val, x_names)

    elif len(x_names) > 1:
        assert len(theta) == len(x_names), f"provide enough names ({x_names}) for the parameters (len{len(theta)})"
        lp = np.sum([log_prior(*_x) for _x in zip(theta, x_names)])
    else:
        raise TypeError(f"""incorrect format provided. Theta should be array-like for single value of x_names or
            Theta should be matrix-like for array-like x_names. Theta, x_names (provided) = {theta, x_names}""")
    if not np.isfinite(lp):
        return -np.inf
    # if type(x_names) == str and x_names == 'log_mass':
    #     model = eval_log_likelihood_mass(theta)
    # else:
    model = eval_log_likelihood(theta, x_names, spectrum_class=DetectorSpectrum)

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
    if x < 0:
        print(f"finding a negative value for {x_name}, returning -np.inf")
        return -np.inf
    if priors[x_name]['prior_type'] == 'flat':
        if 'log' in x_name:
            return log_flat(np.log10(x), x_name)
        else:
            return log_flat(x, x_name)
    elif priors[x_name]['prior_type'] == 'gauss':
        return log_gauss(x, x_name)
    else:
        raise TypeError(f"unknown prior type '{priors[x_name]['prior_type']}', choose either gauss "
                        f"or flat")

# # TODO remove this function
# def eval_log_likelihood_mass(theta):
#     x = theta
#     if np.shape(theta) == (1,):
#         x = theta[0]
#     return GenSpectrum(x, 10e-45, use_SHM, detectors['Xe']).get_data(poisson=False)

def check_shape(xs):
    if not len(xs) > 0:
        raise TypeError(f"Provided incorrect type of {xs}. Takes either np.array or list")
    if not type(xs) == np.array:
        xs = np.array(xs)
    for i, x in enumerate(xs):
        if np.shape(x) == (1,):
            xs[i] = x[0]
    return xs

def eval_log_likelihood(theta, x_names, spectrum_class=GenSpectrum):
    default_order = ['log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density', 'k']
    if len(x_names) == 2:
        x0, x1 = check_shape(theta)
        # if np.shape(x0) == (1,):
        #     x0 = x0[0]
        # if np.shape(x1) == (1,):
        #     x1 = x1[0]
        if x_names[0] == 'log_mass' and x_names[1] == 'log_cross_section':
            return spectrum_class(x0, x1, use_SHM, detectors['Xe']).get_data(poisson=False)
        elif x_names[1] == 'log_mass' and x_names[0] == 'log_cross_section':
            return spectrum_class(x1, x0, use_SHM, detectors['Xe']).get_data(poisson=False)
    elif len(x_names) == 5 or len(x_names) == 6:
        if not x_names == default_order[:len(x_names)]:
            raise NameError(f"The parameters are not input in the correct order. Please insert"
                            f"{default_order[:len(x_names)]} rather than {x_names}.")
        xs = check_shape(theta)
        if len(x_names) == 5:
            fit_shm = SHM(v_0=xs[2]    * nu.km / nu.s,
                          v_esc=xs[3]  * nu.km / nu.s,
                          rho_dm=xs[4] * nu.GeV / nu.c0 ** 2 / nu.cm ** 3)
        if len(x_names) == 6:
            raise NotImplementedError(f"Currently not yet ready to fit for {x_names[5]}")

        result = spectrum_class(xs[0], xs[1], fit_shm, detectors['Xe']).get_data(poisson=False)
        mask = (result['counts'] < 0) & (result['counts'] > -1)
        # TODO this is not the correct way, why does wimprates produce negative rates?
        if np.any(mask):
            print('Serious error, finding negative rates. Presumably v_esc is too small')
            result['counts'][mask] = 0
        return result

    elif len(x_names)>2 and not len(x_names) == 5 and not len(x_names) == 6:
        raise NotImplementedError(f"Not so quick cow-boy, before you code fitting {len(x_names)} "
                                  f"parameters or more, first code it! You are now trying to fit "
                                  f"{x_names}. Make sure that you are not somehow forcing a string "
                                  f"in this part of the code)")
    else:
        raise NotImplementedError(f"Oops this is not somewhere you want to be, x_names = {x_names}")

