"""For a given detector get a WIMPrate for a given detector (not taking into
account any detector effects"""

import numpy as np
import pandas as pd
import wimprates as wr
import numericalunits as nu
from .utils import get_verne_folder, check_folder_for_file
import os
from scipy.interpolate import interp1d
import sys

# # be able to load from the verne folder using this work around.
# sys.path.insert(1, get_verne_folder()+'/src/')
# # python-file in verne folder
# import CalcVelDist_per_v

# VBOUND_MIN = 0 * (nu.km /nu.s)
# VBOUND_MAX = 1000 * (nu.km /nu.s)

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
        self.mw = mw # note that this is not in log scale!
        self.sigma_nucleon = sig # note that this is not in log scale!
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
            # TODO
            #  This integration takes a long time, hence, we will lower the
            #  default precision of the scipy dblquad integration
            migdal_integration_kwargs = dict(epsabs=1e-4,
                                             epsrel=1e-4)
            # TODO this is nasty, we have to circumvent this hardcode
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
                **migdal_integration_kwargs
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
        result = self.set_negative_to_zero(result)
        return result

    @staticmethod
    def set_negative_to_zero(result):
        mask = result['counts'] < 0
        if np.any(mask):
            print('\n\n----\nWARNING::\nfinding negative rates. Doing hard override!!\n----\n\n')
            result['counts'][mask] = 0
            return result
        else:
            return result


class SHM:
    """
        class used to pass a halo model to the rate computation
        must contain:
        :param v_esc -- escape velocity (multiplied by units)
        :param rho_dm -- density in mass/volume of dark matter at the Earth (multiplied by units)
        The standard halo model also allows variation of v_0
        :param v_0 -- v0 of the velocity distribution (multiplied by units)
        :function velocity_dist -- function taking v,t giving normalised
        velocity distribution in earth rest-frame.
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
        class used to pass a halo model to the rate computation based on the
        earth shielding effect as calculated by Verne
        must contain:
        :param v_esc -- escape velocity (multiplied by units)
        :param rho_dm -- density in mass/volume of dark matter at the Earth (multiplied by units)
        The standard halo model also allows variation of v_0
        :param v_0 -- v0 of the velocity distribution (multiplied by units)
        :function velocity_dist -- function taking v,t giving normalised
        velocity distribution in earth rest-frame.
       """

    def __init__(self, v_0=None, v_esc=None, rho_dm=None,
                 log_cross_section=None, log_mass=None, location=None):
        # This may seem somewhat counter intuitive. But we want to use similar
        # input as to SHM (see above) to that end, we here divide the input by
        # the respective units
        self.v_0_nodim = 230 if v_0 is None else v_0 / (nu.km / nu.s)
        self.v_esc_nodim = 544 if v_esc is None else v_esc / (nu.km / nu.s)
        self.rho_dm_nodim = 0.3 if rho_dm is None else rho_dm / (nu.GeV / nu.c0 ** 2 / nu.cm ** 3)

        # Here we keep the units dimensionfull as these parameters are requested
        # by wimprates and therefore must have dimensions
        self.v_0 = self.v_0_nodim * nu.km / nu.s
        self.v_esc = self.v_esc_nodim * nu.km / nu.s
        self.rho_dm = self.rho_dm_nodim * nu.GeV / nu.c0 ** 2 / nu.cm ** 3

        # in contrast to the SHM, the earth shielding does need the mass and
        # cross-section to calculate the rates.
        self.log_cross_section = -35 if log_cross_section is None else log_cross_section
        self.log_mass = 0 if log_mass is None else log_mass
        self.location = "XENON" if location is None else location

        # Combine the parameters into a single naming convention. This is were
        # we will save/read the velocity distribution (from).
        self.fname = 'f_params/loc_%s/v0_%i/vesc_%i/rho_%.3f/sig_%.1f_mx_%.2f' % (
            self.location,
            self.v_0_nodim, self.v_esc_nodim, self.rho_dm_nodim,
            self.log_cross_section, self.log_mass)

        # TODO
        #  Temporary check that these parameters are in a reasonable range.
        assert_str = "double check these parameters"
        for i, param in enumerate([self.v_esc_nodim, self.v_0_nodim, self.rho_dm_nodim]):
            ref_val = [230, 544, 0.3][i]
            # values should be comparable to the reference value
            assert (abs((ref_val - param) / ref_val) < 5 and
                    abs((ref_val - param) / param) < 5), assert_str + f'\nparameter is {param} vs. ref val of {ref_val}'
        self.itp_func = None

    def load_f(self):
        '''
        load the velocity distribution. If there is no velocity distribution shaved, load one.
        :return:
        '''

        # set up folders and names
        folder = get_verne_folder() + 'results/veldists/'
        # TODO
        #  This is a statement to get the data faster, i.e. take a short-cut (we
        #  only compute 2 angles and take the average)
        low_n_gamma = False
        if low_n_gamma:
            self.fname = 'tmp_' + self.fname
        file_name = folder + self.fname + '_avg' + '.csv'
        check_folder_for_file(folder + self.fname)

        # if no data available here, we need to make it
        if not os.path.exists(file_name):
            pyfile = '/src/CalcVelDist.py'
            args = f'-m_x {10 ** self.log_mass} ' \
                   f'-sigma_p {10 ** self.log_cross_section} ' \
                   f'-loc {self.location} ' \
                   f'-path "{get_verne_folder()}/src/" ' \
                   f'-v_0 {self.v_0_nodim} ' \
                   f'-v_esc {self.v_esc_nodim} ' \
                   f'-save_as "{file_name}" '
            if low_n_gamma:
                # Set N_gamma low for faster computation (only two angles)
                args += f' -n_gamma 2'

            cmd = f'python "{get_verne_folder()}"{pyfile} {args}'
            print(f'No spectrum found at:\n{file_name}\nGenerating spectrum, '
                  f'this can take a minute. Execute:\n{cmd}')
            os.system(cmd)
        else:
            print(f'Using {file_name} for the velocity distribution')

        # Alright now load the data and interpolate that. This is the output that wimprates need
        df = pd.read_csv(file_name)
        x, y = df.keys()
        interpolation = interp1d(df[x] * (nu.km / nu.s), df[y] * (nu.s / nu.km), bounds_error=False, fill_value=0)

        # Wimprates needs to have a two-parameter function. However since we
        # ignore time for now. We make this makeshift transition from a one
        # parameter function to a two parameter function

        def velocity_dist(v_, t_):
            return interpolation(v_)

        self.itp_func = velocity_dist

    def velocity_dist(self, v, t):
        # in units of per velocity,
        # v is in units of velocity
        if self.itp_func == None:
            self.load_f()
        return self.itp_func(v, t)


# class ContinuousVerneSHM:
#     """
#         class used to pass a halo model to the rate computation based on the
#         earth shielding effect as calculated by Verne
#         must contain:
#         :param v_esc -- escape velocity (multiplied by units)
#         :param rho_dm -- density in mass/volume of dark matter at the Earth (multiplied by units)
#         The standard halo model also allows variation of v_0
#         :param v_0 -- v0 of the velocity distribution (multiplied by units)
#         :function velocity_dist -- function taking v,t giving normalised
#         velocity distribution in earth rest-frame.
#        """

#     def __init__(self, v_0=None, v_esc=None, rho_dm=None,
#                  log_cross_section=None, log_mass=None, location=None):
#         # This may seem somewhat counter-intuitive. But we want to use similar
#         # input as to SHM (see above) to that end, we here divide the input by
#         # the respective units
#         self.v_0_nodim = 230 if v_0 is None else v_0 / (nu.km / nu.s)
#         self.v_esc_nodim = 544 if v_esc is None else v_esc / (nu.km / nu.s)
#         self.rho_dm_nodim = 0.3 if rho_dm is None else rho_dm / (nu.GeV / nu.c0 ** 2 / nu.cm ** 3)

#         # Here we keep the units dimentionfull as these paramters are requested
#         # by wimprates and therefore must have dimensions
#         self.v_0 = self.v_0_nodim * nu.km / nu.s
#         self.v_esc = self.v_esc_nodim * nu.km / nu.s
#         self.rho_dm = self.rho_dm_nodim * nu.GeV / nu.c0 ** 2 / nu.cm ** 3

#         # in contrast to the SHM, the earth shielding does need the mass and
#         # cross-section to calculate the rates.
#         self.log_cross_section = -35 if log_cross_section is None else log_cross_section
#         self.log_mass = 0 if log_mass is None else log_mass
#         self.location = "XENON" if location is None else location

#         # # Combine the parameters into a single naming convention. This is were
#         # # we will save/read the velocity distributuion (from).
#         # self.fname = 'f_params/loc_%s/v0_%i/vesc_%i/rho_%.2f/sig_%.1f_mx_%.2f' % (
#         #     self.location,
#         #     self.v_0_nodim, self.v_esc_nodim, self.rho_dm_nodim,
#         #     self.log_cross_section, self.log_mass)
#         #
#         # # TODO
#         # #  Temporary check that these parameters are in a reasonable range.
#         assert_str = "double check these parameters"
#         for i, param in enumerate([self.v_esc_nodim, self.v_0_nodim, self.rho_dm_nodim]):
#             ref_val = [230, 544, 0.3][i]
#             # values should be comparable to the reference value
#             assert (abs((ref_val - param) / ref_val) < 5 and
#                     abs((ref_val - param) / param) < 5), assert_str + f'\nparameter is {param} vs. ref val of {ref_val}'
#         # self.itp_func = None

#     # def load_f(self):
#     #     '''
#     #     load the velocity distribution. If there is no velocity distribution shaved, load one.
#     #     :return:
#     #     '''
#     #
#     #     # set up folders and names
#     #     folder = get_verne_folder() + 'results/veldists/'
#     #     # TODO
#     #     #  This is a statement to get the data faster, i.e. take a short-cut (we
#     #     #  only compute 2 angles and take the average)
#     #     low_n_gamma = True
#     #     if low_n_gamma:
#     #         self.fname = 'tmp_' + self.fname
#     #     file_name = folder + self.fname + '_avg' + '.csv'
#     #     check_folder_for_file(folder + self.fname)
#     #
#     #     # if no data available here, we need to make it
#     #     if not os.path.exists(file_name):
#     #         pyfile = '/src/CalcVelDist.py'
#     #         args = f'-m_x {10 ** self.log_mass} ' \
#     #                f'-sigma_p {10 ** self.log_cross_section} ' \
#     #                f'-loc {self.location} ' \
#     #                f'-path "{get_verne_folder()}/src/" ' \
#     #                f'-v_0 {self.v_0_nodim} ' \
#     #                f'-v_esc {self.v_esc_nodim} ' \
#     #                f'-save_as "{file_name}" '
#     #         if low_n_gamma:
#     #             # Set N_gamma low for faster computation (only two angles)
#     #             args += f' -n_gamma 2'
#     #
#     #         cmd = f'python "{get_verne_folder()}"{pyfile} {args}'
#     #         print(f'No spectrum found at:\n{file_name}\nGenerating spectrum, '
#     #               f'this can take a minute. Execute:\n{cmd}')
#     #         os.system(cmd)
#     #     else:
#     #         print(f'Using {file_name} for the velocity distribution')
#     #
#     #     # Alright now load the data and interpolate that. This is the output that wimprates need
#     #     df = pd.read_csv(file_name)
#     #     x, y = df.keys()
#     #     interpolation = interp1d(df[x] * (nu.km / nu.s), df[y] * (nu.s / nu.km), bounds_error=False, fill_value=0)
#     #
#     #     # Wimprates needs to have a two-parameter function. However since we
#     #     # ignore time for now. We make this makeshift transition from a one
#     #     # parameter function to a two parameter function
#     #     def velocity_dist(v_, t_):
#     #         return interpolation(v_)
#     #
#     #     self.itp_func = velocity_dist


#     def load_dist(self, v):
#         low_n_gamma = True
#         distribution_class = CalcVelDist_per_v.CalcVelDist(
#             m_x=10 ** self.log_mass,
#             sigma_p=10 ** self.log_cross_section,
#             path=get_verne_folder()+'/src/',
#             location=self.location,
#             v_0=self.v_0_nodim,
#             v_esc=self.v_esc_nodim,
#             n_gamma=2 if low_n_gamma else 11)

#         return distribution_class.get_f(v)

#     def velocity_dist(self, v, t):
#         # in units of per velocity,
#         # v is in units of velocity

#         def return_velocity_dist(v, t):
#             # TODO
#             #  for now, the time is not taken into account, hence, this work
#             #  around does add a mock-up dependency of the distribution to t.
#             #  This is an option in verne and can be implemented.
#             return self.load_dist(v)

#         return return_velocity_dist(v, t)