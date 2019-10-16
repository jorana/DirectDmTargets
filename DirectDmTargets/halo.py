import numpy as np
import pandas as pd
import wimprates as wr
from scipy.integrate import quad as scipy_int

import numericalunits as nu


#
# def empty_model():
#     return None
#
# config = {
#     'v_0'  : 230  * un.km / un.s,
#     'v_esc': 544  * un.km / un.s,
#     'rho_0': 0.3  * un.GeV / (un.c0**2),
#     'm_dm' : 100  * un.GeV / (un.c0**2),
#     'k'    : 1
# }
#
# # TO Do also have @export?
# # @export
# class dm_halo:
#     """Dark matter halo model. Takes astrophysical parameters and returns the elastic recoil spectrum"""
#
#     def __init__(self):
#         self.v_0 = None
#         self.v_e = None
#         self.v_lsr = None
#         self.model = None
#
#     def model(self, name):
#         # TO DO
#         # Add an assertion error here to check if there is a model
#         assert True
#         model = empty_model
#         return model()
#
def bin_edges(a, b, n):
    """

    :param a: lower limit
    :param b: upper limit
    :param n: number of bins
    :return: bin edges for n bins

    """
    _, edges = np.histogram(np.linspace(a, b), bins=n)
    return edges


def get_bins(a, b, n):
    """
    :param a: lower limit
    :param b: upper limit
    :param n: number of bins
    :return: center of bins
    """
    result = np.vstack((bin_edges(a, b, n)[0:-1], bin_edges(a, b, n)[1:]))
    return np.transpose(result)


class GenSpectrum:
    def __init__(self, mw, sig, model, det):
        """
        :param mw: wimp mass
        :param sig: crossection of the wimp nucleon interaction
        :param model: the dark matter model
        :param det: detector name
        """
        self.mw = mw
        self.sigma_nucleon = sig
        self.dm_model = model
        self.detector = det

        self.n_bins = 10
        self.E_min = 0
        self.E_max = 100


    def __str__(self):
        """
        :return: info
        """
        return f"""spectrum of a DM model ({self.dm_model}) in a {self.detector} detector"""

    def get_bin_centers(self):
        return np.mean(get_bins(self.E_min, self.E_max, self.n_bins), axis=1)

    def spectrum(self, benchmark):
        """
        :param benchmark: insert the kind of DM to consider (should contain Mass and Crossection
        :return: returns the rate
        """
        if (not type(benchmark) == dict) or (not type(benchmark) == pd.DataFrame):
            benchmark = {'mw': benchmark[0],
                         'sigma_nucleon': benchmark[1]}

        # Not normalized
        rate = wr.rate_wimp_std(self.get_bin_centers(),
                                benchmark["mw"],
                                benchmark["sigma_nucleon"],
                                halo_model=self.dm_model
                                )
        return rate

    def get_events(self):
        """
        :return: Events (binned)
        """
        assert self.detector != {}, "First enter the parameters of the detector"
        rate = self.spectrum([self.mw, self.sigma_nucleon])
        bin_width = np.diff(get_bins(self.E_min, self.E_max, self.n_bins), axis=1)[:, 0]
        events = rate * bin_width * self.detector['exp_eff']
        return events

    def get_poisson_events(self):
        """
        :return: events with poisson noise
        """
        return np.random.poisson(self.get_events())

    def get_data(self, poisson=True):
        """

        :param poisson: type bool, add poisson True or False
        :return: pd.DataFrame containing events binned in energy
        """
        result = pd.DataFrame()
        if poisson:
            result['counts'] = self.get_poisson_events()
        else:
            result['counts'] = self.get_events()
        result['bin_centers'] = self.get_bin_centers()
        bins = get_bins(self.E_min, self.E_max, self.n_bins)
        result['bin_left'] = bins[:, 0]
        result['bin_right'] = bins[:, 1]
        return result


def test_test():
    print('done')


# @export
class SHM:
    """
        class used to pass a halo model to the rate computation
        must contain:
        :param v_esc -- escape velocity
        :function velocity_dist -- function taking v,t
        giving normalised valocity distribution in earth rest-frame.
        :param rho_dm -- density in mass/volume of dark matter at the Earth
        The standard halo model also allows variation of v_0
        :param v_0
    """

    def __init__(self, v_0=None, v_esc=None, rho_dm=None):
        self.v_0 = 230 * nu.km / nu.s if v_0 is None else v_0
        self.v_esc = 544 * nu.km / nu.s if v_esc is None else v_esc
        self.rho_dm = 0.3 * nu.GeV / nu.c0 ** 2 / nu.cm ** 3 if rho_dm is None else rho_dm

    def velocity_dist(self, v, t):
        # in units of per velocity,
        # v is in units of velocity
        return wr.observed_speed_dist(v, t, self.v_0, self.v_esc)


# @export
class SHM_12:
    """
        class used to pass a halo model to the rate computation
        must contain:
        :param v_esc -- escape velocity
        :function velocity_dist -- function taking v,t
        giving normalised valocity distribution in earth rest-frame.
        :param rho_dm -- density in mass/volume of dark matter at the Earth
        The standard halo model also allows variation of v_0
        :param v_0
    """

    def __init__(self, v_0=None, v_esc=None, rho_dm=None):
        self.v_0 = 230 * nu.km / nu.s if v_0 is None else v_0
        self.v_esc = 544 * nu.km / nu.s if v_esc is None else v_esc
        self.rho_dm = 0.3 * nu.GeV / nu.c0 ** 2 / nu.cm ** 3 if rho_dm is None else rho_dm

    def velocity_dist(self, v, t):
        # in units of per velocity,
        # v is in units of velocity
        return wr.observed_speed_dist(v, t, self.v_0, self.v_esc)
