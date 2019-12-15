"""For a given detector get a WIMPrate for a given detector (not taking into
account any detector effects"""

import numpy as np
import pandas as pd
import wimprates as wr
import numericalunits as nu
from .utils import get_verne_folder
import os
from scipy.interpolate import interp1d
VBOUND = 10000 * (nu.km /nu.s)

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
        self.experiment = det

        self.n_bins = 10
        if self.experiment['type'] == 'SI':
            self.E_min = 0  # keV
            self.E_max = 100  # keV
        elif self.experiment['type'] == 'migdal':
            self.E_min = 0  # keV
            self.E_max = 10  # keV
        assertion_string = "temporary assertion statement to check that the " \
                           "mass and the cross-section do not go beyond the " \
                           "boundaries of the prior."
        # assert 1 <= mw <= 1000, assertion_string
        # assert 1e-49 <= sig <= 1e-42, assertion_string

    def __str__(self):
        """
        :return: info
        """
        return f"spectrum_simple of a DM model ({self.dm_model}) in a " \
               f"{self.experiment['name']} detector"

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
            kwargs = {'material': self.experiment['material']}
        except KeyError as e:
            print(self.experiment)
            raise e
        if self.experiment['type'] == 'SI':
            rate = wr.rate_wimp_std(self.get_bin_centers(),
                                    benchmark["mw"],
                                    benchmark["sigma_nucleon"],
                                    halo_model=self.dm_model,
                                    **kwargs
                                    )
        elif self.experiment['type'] == 'migdal':
            #TODO this is nasty, we have to circumvent this hardcode
            convert_units = (nu.keV * (1000 * nu.kg) * nu.year)
            rate = convert_units * wr.rate_migdal(
                self.get_bin_centers() * nu.keV,
                benchmark["mw"] * nu.GeV / nu.c0 ** 2,
                benchmark["sigma_nucleon"] * nu.cm ** 2,
                # TODO should this be different for the different experiments?
                q_nr=0.15,
                #
                halo_model=self.dm_model,
                material=self.experiment['material'],
            )
        else:
            raise NotImplementedError(f'No type of matching {self.experiment["type"]} interactions.')
        return rate

    def get_events(self):
        """
        :return: Events (binned)
        """
        assert self.experiment != {}, "First enter the parameters of the detector"
        rate = self.spectrum_simple([self.mw, self.sigma_nucleon])
        bin_width = np.diff(get_bins(self.E_min, self.E_max, self.n_bins),
                            axis=1)[:, 0]
        events = rate * bin_width * self.experiment['exp_eff']
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

class VerneSHM:
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
    def __init__(self, v_0=None, v_esc=None, rho_dm=None, log_cross_section = None,
                 log_mass = None, location = None):
        self.v_0 = 230 if v_0 is None else v_0
        self.v_esc = 544 if v_esc is None else v_esc
        self.rho_dm = 0.3 if rho_dm is None else rho_dm
        self.log_cross_section = -35 if log_cross_section is None else log_cross_section
        self.log_mass = 0 if log_mass is None else log_mass
        self.location = "XENON" if location is None else location
        self.fname = 'f_params_%s_%i_%i_%.2f_%.1f_%.2f'%(
            self.location,
            self.v_0, self.v_esc, self.rho_dm,
            self.log_cross_section, self.log_mass)
        # self.load_f()

    def load_f(self):
        '''
        load the velocity distribution. If there is no velocity distribution shaved, load one.
        :return:
        '''
        folder = get_verne_folder() + 'results/veldists/'
        file_name = folder + self.fname + '_avg' + '.csv'

        if not os.path.exists(file_name):
            pyfile = '/src/CalcVelDist.py'
            args = f'-m_x {10**self.log_mass} -sigma_p {10**self.log_cross_section} -loc {self.location} ' \
                   f'-path "{get_verne_folder()}/src/" -v_0 {self.v_0} -v_esc {self.v_esc} ' \
                   f'-save_as "{file_name}"'
            cmd = f'python "{get_verne_folder()}"{pyfile} {args}'
            print(f'generating spectrum, this can take a minute. Execute:\n{cmd}')
            os.system(cmd)

        df = pd.read_csv(file_name)
        x, y = df.keys()
        # # interpolation = interp1d(df[x] * (nu.km /nu.s), df[y] * (nu.s/nu.km))
        # df[x] * (nu.km / nu.s)
        # df[y]

        df.loc[len(df)] = [-VBOUND, 0]
        df.loc[len(df)+1] = [+VBOUND, 0]
        df.sort_values(by=[x])
        interpolation = interp1d(df[x] * (nu.km /nu.s), df[y] * (nu.s/nu.km))

        def velocity_dist(v_, t_):
            try:
                return interpolation(v_)
            except ValueError:
                print(v_)
                # exit(-1)

        return velocity_dist

    def velocity_dist(self, v, t):
        # in units of per velocity,
        # v is in units of velocity
        return self.load_f()(v, t)
