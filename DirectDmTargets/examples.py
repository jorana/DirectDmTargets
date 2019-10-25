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


def plt_ll_sigma():
    use_SHM = SHM()
    xe_events = GenSpectrum(50, 10e-45, use_SHM, detectors['Xe'])
    xe_data = xe_events.get_data(poisson=True)
    sigma_nucleon = np.linspace(0.1 * 10e-45, 10 * 10e-45, 30)
    model = lambda x: GenSpectrum(50, x, use_SHM, detectors['Xe']).get_data(poisson=False)
    plr = [log_likelihood_df(model(x), xe_data) for x in tqdm(sigma_nucleon)]
    plt.scatter(sigma_nucleon, plr)
    plt.axvline(10e-45)
    plt.xlim(sigma_nucleon[0], sigma_nucleon[-1])
    plt.ylim(np.min(plr), np.max(plr))


def plt_ll_mass():
    use_SHM = SHM()
    mass = np.linspace(5, 100, 100)
    xe_events = GenSpectrum(50, 10e-45, use_SHM, detectors['Xe'])
    # TODO should add the poissonian noise
    xe_data = xe_events.get_data(poisson=False)

    model = lambda x: GenSpectrum(x, 10e-45, use_SHM, detectors['Xe']).get_data(poisson=False)
    plr = [log_likelihood_df(model(x), xe_data) for x in tqdm(mass)]

    mass, plr = remove_nan(mass, plr), remove_nan(plr, mass)
    assert len(mass) > 0, "empty data remains"

    plt.scatter(mass, plr)
    plt.axvline(50)
    # plt.xlim(sigma_nucleon[0], sigma_nucleon[-1])

    plt.ylim(np.min(plr), np.max(plr))


def plt_priors():
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
