"""For a given detector get a WIMPrate for a given detector (not taking into
account any detector effects"""

from warnings import warn
from DirectDmTargets.context import tmp_folder, context
from DirectDmTargets import utils
import numpy as np
import pandas as pd
import wimprates as wr
import numericalunits as nu
import os
from scipy.interpolate import interp1d
import datetime
import time
import subprocess
import verne


def file_ready(name, cmd, max_time=30, max_age=300):
    """
    Check the file is ready when we execute cmd
    Author: A. Pickford
    :param name: name of the file that is to be written
    :param cmd: the command used to create that file
    :param max_time: max. minutes this process waits for the file to be written
    :param max_age: max. age (in minutes) of the file, if the file is older than this, remove it
    :return: is the file written within max time. type(bool)
    """
    print('file_ready: start')
    endTime = datetime.datetime.now() + datetime.timedelta(minutes=max_time)
    flagName = '{0}.flag'.format(name)
    if (os.path.exists(flagName) and
            time.time() - os.path.getmtime(flagName) > max_age * 60):
        print(f'file_ready: found old flag {flagName}. Remove it')
        os.remove(flagName)
    print('file_ready: begin while loop')
    while datetime.datetime.now() < endTime:
        if os.path.exists(name):
            print('file_ready: file exists')
            # file exists, check for flag file
            if os.path.exists(flagName):
                print('file_ready: flag file exists')
                # file and flag file both exist, another process should be
                # creating the file, so wait 30 seconds for other process
                # to finish and delete flag file then retry file checks
                time.sleep(30)
                continue
            else:
                print('file_ready: flag file does not exist')
                # file and exists and no flag file, all is good use the file
                return True
        else:
            print('file_ready: file does not exist')
            # file does not exist, try and make the flag file
            try:
                with open(flagName, 'w') as flag:
                    flag.write('0\n')
            except IOError as e:
                # error creating flag file, most likely another process has just
                # opened the file, this relies on dcache throwing us an
                # IOError back if we try to write to an existing file so in a race
                # to write the file someone is first and someone gets the error
                # we got the error so wait 30 seconds and retry the file checks
                print('file_ready: error creating flag file')
                time.sleep(30)
                continue
            # we wrote the flag file and should now create the real file
            # execute 'cmd' to generate the file
            print(f'file_ready: exec {cmd}')
            subprocess.call(cmd, shell=True)
            print('file_ready: flag file created')
            print('file_ready: file write end')

            # delete flag file
            print('file_ready: delete flag file')
            os.remove(flagName)
            print('file_ready: end true')
            return True

    # if the file isn't ready after maxtime minutes give up and return false
    print('file_ready: end false')
    return False


class GenSpectrum:
    def __init__(self, mw, sig, model, det):
        """
        :param mw: wimp mass
        :param sig: cross-section of the wimp nucleon interaction
        :param model: the dark matter model
        :param det: detector name
        """
        assert isinstance(
            det, dict), "Invalid detector type. Please provide dict."
        self.mw = mw  # note that this is not in log scale!
        self.sigma_nucleon = sig  # note that this is not in log scale!
        self.dm_model = model
        self.experiment = det

        self.n_bins = 10
        if self.experiment['type'] in ['SI', 'SI_bg']:
            self.E_min = 0  # keV
            self.E_max = 100  # keV
        elif self.experiment['type'] in ['migdal', 'migdal_bg']:
            self.E_min = 0  # keV
            self.E_max = 10  # keV
        else:
            raise NotImplementedError(
                f'Exp. type {self.experiment["type"]} is unknown')

        if 'E_min' in self.experiment:
            self.E_min = self.experiment['E_min']
        if 'E_max' in self.experiment:
            self.E_max = self.experiment['E_max']

    def __str__(self):
        """
        :return: info
        """
        return f"spectrum_simple of a DM model ({self.dm_model}) in a " \
               f"{self.experiment['name']} detector"

    def get_bin_centers(self):
        return np.mean(
            utils.get_bins(
                self.E_min,
                self.E_max,
                self.n_bins),
            axis=1)

    def spectrum_simple(self, benchmark):
        """
        :param benchmark: insert the kind of DM to consider (should contain Mass
         and cross-section)
        :return: returns the rate
        """
        if not isinstance(benchmark, (dict, pd.DataFrame)):
            benchmark = {'mw': benchmark[0], 'sigma_nucleon': benchmark[1]}

        try:
            kwargs = {'material': self.experiment['material']}
        except KeyError as e:
            print(self.experiment)
            raise e
        if self.experiment['type'] in ['SI', 'SI_bg']:
            rate = wr.rate_wimp_std(self.get_bin_centers(),
                                    benchmark["mw"],
                                    benchmark["sigma_nucleon"],
                                    halo_model=self.dm_model,
                                    **kwargs
                                    )
        elif self.experiment['type'] in ['migdal', 'migdal_bg']:
            # TODO
            #  This integration takes a long time, hence, we will lower the
            #  default precision of the scipy dblquad integration
            migdal_integration_kwargs = dict(epsabs=1e-4,
                                             epsrel=1e-4)
            convert_units = (nu.keV * (1000 * nu.kg) * nu.year)
            rate = convert_units * wr.rate_migdal(
                self.get_bin_centers() * nu.keV,
                benchmark["mw"] * nu.GeV / nu.c0 ** 2,
                benchmark["sigma_nucleon"] * nu.cm ** 2,
                # TODO should this be different for the different experiments?
                q_nr=0.15,
                halo_model=self.dm_model,
                material=self.experiment['material'],
                **migdal_integration_kwargs
            )
        else:
            raise NotImplementedError(
                f'No type of matching {self.experiment["type"]} interactions.')
        return rate

    def get_events(self):
        """
        :return: Events (binned)
        """
        assert self.experiment != {}, "First enter the parameters of the detector"
        rate = self.spectrum_simple([self.mw, self.sigma_nucleon])
        bin_width = np.diff(
            utils.get_bins(
                self.E_min,
                self.E_max,
                self.n_bins),
            axis=1)[
            :,
            0]
        events = rate * bin_width * self.experiment['exp_eff']
        return events

    def get_poisson_events(self):
        """
        :return: events with poisson noise
        """
        return np.random.exponential(self.get_events()).astype(np.float)

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
        bins = utils.get_bins(self.E_min, self.E_max, self.n_bins)
        result['bin_left'] = bins[:, 0]
        result['bin_right'] = bins[:, 1]
        result = self.set_negative_to_zero(result)
        return result

    @staticmethod
    def set_negative_to_zero(result):
        mask = result['counts'] < 0
        if np.any(mask):
            print(
                '\n\n----\nWARNING::\nfinding negative rates. Doing hard override!!\n----\n\n')
            result['counts'][mask] = 0
            return result
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

    def __str__(self):
        # Standard Halo Model (shm)
        return 'shm'

    def velocity_dist(self, v, t):
        """
        Get the velocity distribution in units of per velocity,
        :param v: v is in units of velocity
        :return: observed velocity distribution at earth
        """
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
        self.rho_dm_nodim = 0.3 if rho_dm is None else rho_dm / \
            (nu.GeV / nu.c0 ** 2 / nu.cm ** 3)

        # Here we keep the units dimensionful as these parameters are requested
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

        self.itp_func = None

    def __str__(self):
        # The standard halo model observed at some location shielded from strongly
        # interacting DM by overburden (rock atmosphere)
        return 'shielded_shm'

    def load_f(self):
        """
        load the velocity distribution. If there is no velocity distribution shaved, load one.
        :return:
        """

        # set up folders and names
        file_folder = context['verne_files']
        software_folder = context['verne_folder']
        file_name = os.path.join(file_folder, self.fname + '_avg' + '.csv')
        utils.check_folder_for_file(os.path.join(file_folder, self.fname))

        # Convert file_name and self.fname to folder and name of csv file where
        # to save.
        exist_csv, abs_file_name = utils.add_identifier_to_safe(file_name)
        assertion_string = f'abs file {abs_file_name} should be a string\n'
        assertion_string += f'exists csv {exist_csv} should be a bool'
        assert isinstance(
            abs_file_name, str) and isinstance(
            exist_csv, bool), assertion_string
        if not exist_csv:
            verne.CalcVelDist.write_calcveldist(
                m_x=10. ** self.log_mass,
                sigma_p=10. ** self.log_cross_section,
                loc=self.location,
                v_esc=self.v_esc_nodim,
                v_0=self.v_0_nodim,
                save_as=file_name,
                N_gamma=4,
            )
            # pyfile = os.path.join(verne.__path__[0], 'CalcVelDist.py')
            # file_name = tmp_folder + utils.unique_hash() + '.csv'
            # args = (f'-m_x {10. ** self.log_mass} '
            #         f'-sigma_p {10. ** self.log_cross_section} '
            #         f'-loc {self.location} '
            #         # f'-path "{software_folder}/src/" '
            #         f'-v_0 {self.v_0_nodim} '
            #         f'-v_esc {self.v_esc_nodim} '
            #         f'-save_as "{file_name}"')
            #
            # cmd = f'python {pyfile} {args}'
            # print(f'No spectrum found at:\n{file_name}\nGenerating spectrum, '
            #       f'this can take a minute. Execute:\n{cmd}')
            # assert file_ready(file_name, cmd), f"{file_name} could not be written"
            mv_cmd = f'mv {file_name} {abs_file_name}'
            if not os.path.exists(abs_file_name):
                print(f'load_f:\tcopy from temp-folder to verne_folder')
                file_ready(abs_file_name, mv_cmd, max_time=1)
            else:
                warn(
                    f'load_f:\twhile writing {file_name}, {abs_file_name} was created')
        else:
            print(f'Using {abs_file_name} for the velocity distribution')

        # Alright now load the data and interpolate that. This is the output
        # that wimprates need
        if not os.path.exists(abs_file_name):
            raise OSError(f'{abs_file_name} should exist')
        try:
            df = pd.read_csv(abs_file_name)
        except pd.io.common.EmptyDataError as pandas_error:
            os.remove(abs_file_name)
            raise pandas_error

        if not len(df):
            # Somehow we got an empty dataframe, we cannot continue
            os.remove(abs_file_name)
            raise ValueError(
                f'Was trying to read an empty dataframe from {abs_file_name}')

        x, y = df.keys()
        interpolation = interp1d(
            df[x] * (nu.km / nu.s), df[y] * (nu.s / nu.km), bounds_error=False, fill_value=0)

        def velocity_dist(v_, t_):
            # Wimprates needs to have a two-parameter function. However since we
            # ignore time for now. We make this makeshift transition from a one
            # parameter function to a two parameter function
            return interpolation(v_)

        self.itp_func = velocity_dist

    def velocity_dist(self, v, t):
        """
        Get the velocity distribution in units of per velocity,
        :param v: v is in units of velocity
        :return: observed velocity distribution at earth
        """
        if self.itp_func is None:
            self.load_f()
        return self.itp_func(v, t)
