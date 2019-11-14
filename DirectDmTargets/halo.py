"""For a given detector get a WIMPrate for a given detector (not taking into
account any detector effects"""

import numpy as np
import pandas as pd
import wimprates as wr
import numericalunits as nu


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
        :param sig: cross-section of the wimp nucleon interaction
        :param model: the dark matter model
        :param det: detector name
        """
        assert type(det) is dict, "Invalid detector type. Please provide dict."
        self.mw = mw
        self.sigma_nucleon = sig
        self.dm_model = model
        self.detector = det

        self.n_bins = 10
        self.E_min = 0  # keV
        self.E_max = 100  # keV

        assertion_string = "temporary assertion statement to check that the " \
                           "mass and the cross-section do not go beyond the " \
                           "boundaries of the prior."
        assert 1 <= mw <= 1000, assertion_string
        assert 1e-46 <= sig <= 1e-42, assertion_string

    def __str__(self):
        """
        :return: info
        """
        return f"spectrum_simple of a DM model ({self.dm_model}) in a " \
               f"{self.detector['name']} detector"

    def get_bin_centers(self):
        return np.mean(get_bins(self.E_min, self.E_max, self.n_bins), axis=1)

    def spectrum_simple(self, benchmark):
        """
        :param benchmark: insert the kind of DM to consider (should contain Mass
         and Crossection)
        :return: returns the rate
        """
        if ((not type(benchmark) is dict) or (
                not type(benchmark) is pd.DataFrame)):
            benchmark = {'mw': benchmark[0],
                         'sigma_nucleon': benchmark[1]}

        try:
            kwargs = {'material': self.detector['name']}
        except KeyError as e:
            print(self.detector)
            raise e
        rate = wr.rate_wimp_std(self.get_bin_centers(),
                                benchmark["mw"],
                                benchmark["sigma_nucleon"],
                                halo_model=self.dm_model,
                                **kwargs
                                )
        return rate

    def get_events(self):
        """
        :return: Events (binned)
        """
        assert self.detector != {}, "First enter the parameters of the detector"
        rate = self.spectrum_simple([self.mw, self.sigma_nucleon])
        bin_width = np.diff(get_bins(self.E_min, self.E_max, self.n_bins),
                            axis=1)[:, 0]
        events = rate * bin_width * self.detector['exp_eff']
        return events

    def get_poisson_events(self):
        """
        :return: events with poisson noise
        """
        return np.random.poisson(self.get_events()).astype(np.float)

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


class SHM:
    """
        class used to pass a halo model to the rate computation
        must contain:
        :param v_esc -- escape velocity
        :function velocity_dist -- function taking v,t giving normalised
        velocity distribution in earth rest-frame.
        :param rho_dm -- density in mass/volume of dark matter at the Earth
        The standard halo model also allows variation of v_0
        :param v_0
    """

    def __init__(self, v_0=None, v_esc=None, rho_dm=None):
        self.v_0 = 230 * nu.km / nu.s if v_0 is None else v_0
        self.v_esc = 544 * nu.km / nu.s if v_esc is None else v_esc
        self.rho_dm = (0.3 * nu.GeV / nu.c0 ** 2 / nu.cm ** 3
                       if rho_dm is None else rho_dm)

    def velocity_dist(self, v, t):
        # in units of per velocity,
        # v is in units of velocity
        return wr.observed_speed_dist(v, t, self.v_0, self.v_esc)
