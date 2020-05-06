"""Do a likelihood fit. The class NestedSamplerStatModel is used for fitting applying the bayesian algorithm nestle"""

from __future__ import absolute_import, unicode_literals, print_function
from .halo import *
from .statistics import *
from .context import *
from .utils import *
from datetime import datetime
import json
import os
from scipy import special as spsp
import corner
import matplotlib.pyplot as plt
import shutil


import tempfile

# import logging

def default_nested_save_dir():
    """The name of folders where to save results from the NestedSamplerStatModel"""
    return 'nested'


class NestedSamplerStatModel(StatModel):
    known_parameters = get_param_list()

    def __init__(self, *args):
        StatModel.__init__(self, *args)
        self.tol = 0.1  # Tolerance for sampling
        self.nlive = 1024  # number of live points
        self.sampler = 'nestle'
        self.log_dict = {'did_run': False, 'saved_in': None, 'tmp_dir': None, 'garbage_bin': []}
        self.config['start'] = datetime.now()
        self.config['notes'] = "default"
        self.result = False
        self.fit_parameters = None
        self.set_fit_parameters(['log_mass', 'log_cross_section'])

        self.log.info(f'NestedSamplerStatModel::\tVERBOSE ENABLED')
        self.log.debug(f"NestedSamplerStatModel::\t{now(self.config['start'])}\n\t"
                       f"SUPERVERBOSE ENABLED\n\tyou want to know it all? Here we go sit "
                       f"back and be blown by my output!")

    def set_fit_parameters(self, params):
        self.log.info(f'NestedSamplerStatModel::\tsetting fit'
                      f' parameters to {params}')
        if not type(params) == list:
            raise TypeError("Set the parameter names in a list of strings")
        for param in params:
            if param not in self.known_parameters:
                err_message = f"{param} does not match any of the known parameters try " \
                              f"any of {self.known_parameters}"
                raise NotImplementedError(err_message)
        if not params == self.known_parameters[:len(params)]:
            err_message = f"The parameters are not input in the correct order. Please" \
                          f" insert {self.known_parameters[:len(params)]} rather than {params}."
            raise NameError(err_message)
        self.fit_parameters = params

    def check_did_run(self):
        if not self.log_dict['did_run']:
            self.log.info(f'NestedSamplerStatModel::\tdid not '
                          f'run yet, lets fire it up!')
            if self.sampler == 'nestle':
                self.run_nestle()
            elif self.sampler == 'multinest':
                self.run_multinest()
            else:
                raise NotImplementedError(f'No such sampler as {self.sampler}, perhaps '
                                          f'try nestle or multinest')
        else:
            self.log.info(f'NestedSamplerStatModel::\tdid run')

    def check_did_save(self):
        self.log.info(f'NestedSamplerStatModel::\tdid not save'
                      f' yet, we dont want to lose our results so better do it now')
        if self.log_dict['saved_in'] is None:
            self.save_results()

    def log_probability_nested(self, parameter_vals, parameter_names):
        """

        :param parameter_vals: the values of the model/benchmark considered as the truth
        # :param parameter_values: the values of the parameters that are being varied
        :param parameter_names: the names of the parameter_values
        :return:
        """
        if self.verbose > 1:
            print(f'NestedSamplerStatModel::\tSUPERVERBOSE\tthere we go! Find that log probability')
        evaluated_rate = self.eval_spectrum(parameter_vals, parameter_names)[
            'counts']

        ll = log_likelihood(self.benchmark_values, evaluated_rate)
        if np.isnan(ll):
            raise ValueError(
                f"Returned NaN from likelihood. ll = {ll}")
        self.log.debug(f'NestedSamplerStatModel::\tSUPERVERBOSE\tfound it! returning the log likelihood')
        return ll

    def log_prior_transform_nested(self, x, x_name):
        self.log.debug(f'NestedSamplerStatModel::\tSUPERVERBOSE\tdoing some transformations for nestle/multinest '
                       f'to read the priors')
        if self.config['prior'][x_name]['prior_type'] == 'flat':
            a, b = self.config['prior'][x_name]['param']
            # Prior transform of a flat prior is a simple line.
            return x * (b - a) + a
        elif self.config['prior'][x_name]['prior_type'] == 'gauss':
            # Get the range from the config file
            a, b = self.config['prior'][x_name]['range']
            m, s = self.config['prior'][x_name]['param']

            # Here the prior transform is being constructed and shifted. This may not seem trivial
            # and one is advised to request a notebook where this is explained from the developer(s).
            aprime = spsp.ndtr((a - m) / s)
            bprime = spsp.ndtr((b - m) / s)
            xprime = x * (bprime - aprime) + aprime
            res = m + s * spsp.ndtri(xprime)
            return res
        else:
            err_message = f"unknown prior type '{self.config['prior'][x_name]['prior_type']}', " \
                          f"choose either gauss or flat"
            raise TypeError(err_message)

    def _log_probability_nested(self, theta):
        ndim = len(theta)
        if self.verbose > 1:
            print(f'NestedSamplerStatModel::\tSUPERVERBOSE\tdoing '
                  f'_log_probability_nested for {ndim} parameters'
                  f'\n\t\tooph, what a nasty function to do some transformations behind the scenes')

        return self.log_probability_nested(theta, self.known_parameters[:ndim])

    def _log_prior_transform_nested(self, theta):
        self.log.debug(f'NestedSamplerStatModel::\tSUPERVERBOSE\tdoing '
                       f'_log_prior_transform_nested for {len(theta)} parameters'
                       f'\n\t\tooph, what a nasty function to do some transformations behind the scenes')
        result = [self.log_prior_transform_nested(val, self.known_parameters[i]) for i, val in enumerate(theta)]
        return np.array(result)

    def run_nestle(self):
        assert self.sampler == 'nestle', f'Trying to run nestle but initialization requires {self.sampler}'

        # Do the import of nestle inside the class such that the package can be
        # loaded without nestle
        try:
            import nestle
        except ModuleNotFoundError:
            raise ModuleNotFoundError('package nestle not found. See README for installation')

        if self.verbose:
            print(f'NestedSamplerStatModel::\tWe made it to my core function, lets do that optimization')
        method = 'multi'  # use MutliNest algorithm
        ndim = len(self.fit_parameters)
        tol = self.tol  # the stopping criterion
        self.log.info(f'NestedSamplerStatModel::\there we go! We are going to fit:\n\t{ndim} parameters\n')
        assert_str = f"Unknown configuration of fit pars: {self.fit_parameters}"
        assert self.fit_parameters == self.known_parameters[:ndim], assert_str
        try:
            self.log.warning(f'run_nestle::\tstart_fit for %i parameters' % ndim)
            self.log.info(f'NestedSamplerStatModel::\tbeyond this point, there is nothing '
                          f"I can say, you'll have to wait for my lower level "
                          f'algorithms to give you info, see you soon!')
            start = datetime.now()
            self.result = nestle.sample(self._log_probability_nested,
                                        self._log_prior_transform_nested,
                                        ndim,
                                        method=method,
                                        npoints=self.nlive,
                                        dlogz=tol)
            end = datetime.now()
            dt = end - start
            self.log.info(f'run_nestle::\tfit_done in %i s (%.1f h)' % (dt.seconds, dt.seconds / 3600.))
            self.log.debug(f'NestedSamplerStatModel::\tSUPERVERBOSE\tWe are back!')
        except ValueError as e:
            self.log.error(f'Nestle did not finish due to a ValueError. Was running with'
                           f'\n{len(self.fit_parameters)} for fit ''parameters ' f'{self.fit_parameters}')
            raise e
        self.log_dict['did_run'] = True
        try:
            self.config['fit_time'] = dt.seconds
        except NameError:
            self.config['fit_time'] = -1
        self.log.info(f'NestedSamplerStatModel::\tFinished with running optimizer!')

    def print_before_run(self):
        self.log.info(f"""--------------------------------------------------
        NestedSamplerStatModel::\t{now(self.config['start'])}\n\tFinal print of all of the set options:
        self.tol = {self.tol}
        self.nlive = {self.nlive}
        self.sampler = {self.sampler} 
        self.log = {self.log}
        self.config = {self.config}
        self.result = {self.result}
        self.fit_parameters = {self.fit_parameters}
        halo_model = {self.config['halo_model']} with:
            v_0 = {self.config['halo_model'].v_0 / (nu.km / nu.s)}
            v_esc = {self.config['halo_model'].v_esc / (nu.km / nu.s)}
            rho_dm = {self.config['halo_model'].rho_dm / (nu.GeV / nu.c0 ** 2 / nu.cm ** 3)}
        --------------------------------------------------
        """)

    def run_multinest(self):
        assert self.sampler == 'multinest', f'Trying to run multinest but initialization requires {self.sampler}'
        # Do the import of multinest inside the class such that the package can be
        # loaded without multinest
        try:
            from pymultinest.solve import run, Analyzer, solve
        except ModuleNotFoundError:
            raise ModuleNotFoundError('package pymultinest not found. See README for installation')

        self.log.info(f'NestedSamplerStatModel::\tWe made it to my core function, lets do that optimization')

        n_dims = len(self.fit_parameters)
        tol = self.tol  # the stopping criterion
        self.log.info(f'NestedSamplerStatModel::\there we go! We are going to fit:\n\t{n_dims} parameters\n')
        save_at = self.get_save_dir()
        assert_str = f'Unknown configuration of fit pars: {self.fit_parameters}'
        assert self.fit_parameters == self.known_parameters[:n_dims], assert_str
        # try:
        self.log.warning(f'NestedSamplerStatModel::\tstart_fit for %i parameters' % n_dims)
        self.log.info(f'NestedSamplerStatModel::\tbeyond this point, there is nothing '
                      f"I can say, you'll have to wait for my lower level "
                      f'algorithms to give you info, see you soon!')
        start = datetime.now()

        # Multinest saves output to a folder. First write to the tmp folder, move it to the results folder later
        # _tmp_folder = self.get_tmp_dir()
        _tmp_folder = self.get_save_dir()
        save_at_temp = f'{_tmp_folder}multinest'

        # solve(
        solve_multinest(
            LogLikelihood=self._log_probability_nested,  # SafeLoglikelihood,
            Prior=self._log_prior_transform_nested,  # SafePrior,
            n_live_points=self.nlive,
            n_dims=n_dims,
            outputfiles_basename=save_at_temp,
            verbose=True,
            evidence_tolerance=tol
        )
        self.result = save_at_temp

        # Open a save-folder after successful running multinest. Move the multinest results there.
        # save_at = self.get_save_dir()
        check_folder_for_file(save_at)
        assert _tmp_folder[-1] == '/', 'make sure that tmp_folder ends at "/"'
        copy_multinest = save_at + _tmp_folder.split('/')[-2]
        self.log.info(f'copy {_tmp_folder} to {copy_multinest}')
        self.log_dict['garbage_bin'].append(_tmp_folder)
        end = datetime.now()
        dt = end - start
        self.log.warning(f'run_multinest::\tfit_done in %i s (%.1f h)' % (dt.seconds, dt.seconds / 3600.))

        # except ValueError as e:
        #     self.log.error(
        #         f'Multinest did not finish due to a ValueError. Was running with'
        #         f'\n{len(self.fit_parameters)} for fit parameters {self.fit_parameters}')
        #     raise e
        self.log_dict['did_run'] = True
        try:
            self.config['fit_time'] = dt.seconds
        except NameError:
            self.config['fit_time'] = -1
        self.log.info(f'NestedSamplerStatModel::\tFinished with running Multinest!')

    def empty_garbage(self):
        for file in self.log_dict['garbage_bin']:
            self.log.info(f'delete {file}')
            if os.path.exists(file):
                try:
                    shutil.rmtree(file)
                except FileNotFoundError:
                    pass
            else:
                print(f'Could not find {file} that is in the garbage bin?')

    def get_summary(self):
        self.log.info(f'NestedSamplerStatModel::\tgetting the summary (or at'
                      f"least trying) let's first see if I did run")
        self.check_did_run()
        self.log.info(f"NestedSamplerStatModel::\t{now(self.config['start'])}\n\tAlright, that's done. Let's get some "
                      f"info. I'm not going to print too much here")
        # keep a dictionary of all the results
        resdict = {}

        if self.sampler == 'multinest':
            # Do the import of multinest inside the class such that the package can be
            # loaded without multinest
            try:
                from pymultinest.solve import run, Analyzer, solve
            except ModuleNotFoundError:
                raise ModuleNotFoundError('package pymultinest not found. See README for installation')
            self.log.info('NestedSamplerStatModel::\tget_summary::\tstart analyzer of results')
            analyzer = Analyzer(len(self.fit_parameters), outputfiles_basename=self.result)
            # Taken from multinest.solve
            self.result = analyzer.get_stats()
            samples = analyzer.get_equal_weighted_posterior()[:, :-1]

            self.log.info('parameter values:')
            for name, col in zip(self.fit_parameters, samples.transpose()):
                self.log.info('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
                resdict[name + '_fit_res'] = ('{0:5.2f} +/- {1:5.2f}'.format(col.mean(), col.std()))
                if 'log_' in name:
                    resdict[name[4:] + '_fit_res'] = '%.3g +/- %.2g' % (
                        10. ** col.mean(), 10. ** (col.mean()) * np.log(10.) * col.std())
                    self.log.info(f'\t {name[4:]}, {resdict[name[4:] + "_fit_res"]}')
            resdict['n_samples'] = len(samples.transpose()[0])
            # Pass the samples to the self.result to be saved.
            self.result['samples'] = samples
        elif self.sampler == 'nestle':
            # Do the import of nestle inside the class such that the package can be
            # loaded without nestle
            try:
                import nestle
            except ModuleNotFoundError:
                raise ModuleNotFoundError('package nestle not found. See README for installation')
            # taken from mattpitkin.github.io/samplers-demo/pages/samplers-samplers-everywhere/#Nestle
            # estimate of the statistical uncertainty on logZ
            logZerrnestle = np.sqrt(self.result.h / self.nlive)
            # re-scale weights to have a maximum of one
            nweights = self.result.weights / np.max(self.result.weights)
            # get the probability of keeping a sample from the weights
            keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]
            # get the posterior samples
            samples_nestle = self.result.samples[keepidx, :]
            # estimate of the statistcal uncertainty on logZ
            resdict['nestle_nposterior'] = len(samples_nestle)
            resdict['nestle_time'] = self.config['fit_time']  # run time
            resdict['nestle_logZ'] = self.result.logz  # log marginalised likelihood
            resdict['nestle_logZerr'] = logZerrnestle  # uncertainty on log(Z)
            resdict['summary'] = self.result.summary()
            p, cov = nestle.mean_and_cov(self.result.samples, self.result.weights)
            for i, key in enumerate(self.fit_parameters):
                resdict[key + '_fit_res'] = ('{0:5.2f} +/- {1:5.2f}'.format(p[i], np.sqrt(cov[i, i])))
                self.log.info(f'\t, {key}, {resdict[key + "_fit_res"]}')
                if 'log_' in key:
                    resdict[key[4:] + '_fit_res'] = '%.3g +/- %.2g' % (
                        10. ** p[i], 10. ** (p[i]) * np.log(10) * np.sqrt(cov[i, i]))
                    self.log.info(f'\t, {key[4:]}, {resdict[key[4:] + "_fit_res"]}')
        self.log.info(f'NestedSamplerStatModel::\tAlright we got all the info we need, '
                      f"let's return it to whomever asked for it")
        return resdict

    def get_save_dir(self, force_index=False, hash=None):
        if (not self.log_dict['saved_in']) or force_index:
            self.log_dict['saved_in'] = open_save_dir(f'{default_nested_save_dir()}_{self.sampler}',
                                                      force_index=force_index, hash=hash)
        self.log.info(f'NestedSamplerStatModel::\tget_save_dir\tsave_dir = {self.log_dict["saved_in"]}')
        return self.log_dict['saved_in']

    def get_tmp_dir(self, force_index=False, hash=None):
        if (not self.log_dict['tmp_dir']) or force_index:
            self.log_dict['tmp_dir'] = open_save_dir(f'{self.sampler}', base=context['tmp_folder'],
                                                     force_index=force_index, hash=hash)
        self.log.info(f'NestedSamplerStatModel::\tget_tmp_dir\ttmp_dir = {self.log_dict["tmp_dir"]}')
        return self.log_dict['tmp_dir']

    def save_results(self, force_index=False):
        self.log.info(f'NestedSamplerStatModel::\tAlmost there! We are about to save the '
                      f'results. But first do some checks, did we actually run?')
        # save fit parameters to config
        self.config['fit_parameters'] = self.fit_parameters
        self.config['tol'] = self.tol
        self.check_did_run()
        save_dir = self.get_save_dir(force_index=force_index)
        fit_summary = self.get_summary()
        self.log.info(f'NestedSamplerStatModel::\tAlright all set, let put all that info'
                      f' in {save_dir} and be done with it')
        # save the config, chain and flattened chain
        if 'HASH' in save_dir or os.path.exists(save_dir + 'config.json'):
            save_dir += 'pid' + str(os.getpid()) + '_'
        with open(save_dir + 'config.json', 'w') as file:
            json.dump(convert_dic_to_savable(self.config), file, indent=4)
        with open(save_dir + 'res_dict.json', 'w') as file:
            json.dump(convert_dic_to_savable(fit_summary), file, indent=4)
        np.save(save_dir + 'config.npy', convert_dic_to_savable(self.config))
        np.save(save_dir + 'res_dict.npy', convert_dic_to_savable(fit_summary))
        for col in self.result.keys():
            if col == 'samples' or type(col) is not dict:
                if self.sampler == 'multinest' and col == 'samples':
                    # in contrast to nestle, multinest returns the weighted samples.
                    np.save(save_dir + 'weighted_samples.npy', self.result[col])
                else:
                    np.save(save_dir + col + '.npy', self.result[col])
            else:
                np.save(save_dir + col + '.npy',
                        convert_dic_to_savable(self.result[col]))
        shutil.copy(self.config['logging'], save_dir + self.config['logging'].split('/')[-1])
        self.log.info(f'save_results::\tdone_saving')

    def show_corner(self):
        self.log.info(
            f"NestedSamplerStatModel::\t{now(self.config['start'])}\n\tLet's do some graphics, I'll make you a "
            f"nice corner plot just now")
        self.check_did_save()
        save_dir = self.log_dict['saved_in']
        combined_results = load_nestle_samples_from_file(save_dir)
        nestle_corner(combined_results, save_dir)
        self.log.info(f'NestedSamplerStatModel::\tEnjoy the plot. Maybe you do want to'
                      f' save it too?')


def is_savable_type(item):
    if type(item) in [list, np.array, np.ndarray, int, str, np.int, np.float, bool, np.float64]:
        return True
    return False


def convert_dic_to_savable(config):
    result = config.copy()
    for key in result.keys():
        if is_savable_type(result[key]):
            pass
        elif type(result[key]) == dict:
            result[key] = convert_dic_to_savable(result[key])
        else:
            result[key] = str(result[key])
    return result


def load_nestle_samples(load_from=default_nested_save_dir(), item='latest'):
    base = get_result_folder()
    save = load_from
    files = os.listdir(base)
    if item == 'latest':
        item = max([int(f.split(save)[-1]) for f in files if save in f])

    load_dir = base + save + str(item) + '/'
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f'Cannot find {load_dir} specified by arg: '
                                f'{item}')
    return load_nestle_samples_from_file(load_dir)


def load_nestle_samples_from_file(load_dir):
    print(f'load_nestle_samples::\tloading', load_dir)
    keys = ['config', 'res_dict', 'h', 'logl', 'logvol', 'logz', 'logzerr',
            'ncall', 'niter', 'samples', 'weights']
    result = {}
    for key in keys:
        result[key] = np.load(load_dir + key + '.npy', allow_pickle=True)
        if key == 'config' or key == 'res_dict':
            result[key] = result[key].item()
    print(f"load_nestle_samples::\tdone loading\naccess result with:\n{keys}")
    return result


def load_multinest_samples_from_file(load_dir):
    keys = os.listdir(load_dir)
    keys = [key for key in keys if os.path.isfile(load_dir + '/' + key)]
    result = {}
    for key in keys:
        if '.npy' in key:
            naked_key = key.split('.npy')[0]
            naked_key = do_strip_from_pid(naked_key)
            tmp_res = np.load(load_dir + key, allow_pickle=True)
            if naked_key == 'config' or naked_key == 'res_dict':
                result[naked_key] = tmp_res.item()
            else:
                result[naked_key] = tmp_res
    return result


def do_strip_from_pid(string):
    """
    remove PID identifier from a string
    """
    if 'pid' not in string:
        return string
    else:
        new_key = string.split("_")
        new_key = "_".join(new_key[1:])
        return new_key


def load_multinest_samples(load_from=default_nested_save_dir(), item='latest'):
    base = get_result_folder()
    save = load_from
    files = os.listdir(base)
    if item == 'latest':
        item = max([int(f.split(save)[-1]) for f in files if save in f])

    load_dir = base + save + str(item) + '/'
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Cannot find {load_dir} specified by arg: "
                                f"{item}")
    return load_multinest_samples_from_file(load_dir)


def multinest_corner(result, save=False):
    info = "$M_\chi}$=%.2f" % 10. ** np.float(result['config']['mw'])
    for prior_key in result['config']['prior'].keys():
        try:
            mean = result['config']['prior'][prior_key]['mean']
            info += f"\n{prior_key} = {mean}"
        except KeyError:
            pass
    nposterior, ndim = np.shape(result['weighted_samples'])
    info += "\nnposterior = %s" % nposterior
    for str_inf in ['detector', 'notes', 'start', 'fit_time', 'poisson', 'n_energy_bins']:
        try:
            info += f"\n{str_inf} = %s" % result['config'][str_inf]
            if str_inf == 'start':
                info = info[:-7]
            if str_inf == 'fit_time':
                info += 's (%.1f h)' % (result['config'][str_inf] / 3600.)
        except KeyError:
            # We were trying to load something that wasn't saved in the config file, ignore it for now.
            pass
    labels = get_param_list()[:ndim]
    try:
        truths = [result['config'][prior_name] for prior_name in get_prior_list()[:ndim]]
    except KeyError:
        truths = []
        for prior_name in get_prior_list()[:ndim]:
            if prior_name != "rho_0":
                truths.append(result['config'][prior_name])
            else:
                truths.append(result['config']['density'])
    fig = corner.corner(
        result['weighted_samples'],
        labels=labels,
        range=[0.99999, 0.99999, 0.99999, 0.99999, 0.99999][:ndim],
        truths=truths,
        show_titles=True)
    fig.axes[1].set_title(f"Fit title", loc='left')
    fig.axes[1].text(0, 1, info, verticalalignment='top')
    if save:
        plt.savefig(f"{save}corner.png", dpi=200)
    # plt.show()


def nestle_corner(result, save=False):
    info = "$M_\chi}$=%.2f" % 10. ** np.float(result['config']['mw'])
    for prior_key in result['config']['prior'].keys():
        try:
            mean = result['config']['prior'][prior_key]['mean']
            info += f"\n{prior_key} = {mean}"
        except KeyError:
            pass
    nposterior, ndim = np.shape(result['samples'])
    info += "\nnposterior = %s" % nposterior
    for str_inf in ['detector', 'notes', 'start', 'fit_time', 'poisson', 'n_energy_bins']:
        try:
            info += f"\n{str_inf} = %s" % result['config'][str_inf]
            if str_inf == 'start':
                info = info[:-7]
            if str_inf == 'fit_time':
                info += 's (%.1f h)' % (result['config'][str_inf] / 3600.)
        except KeyError:
            # We were trying to load something that wasn't saved in the config file, ignore it for now.
            pass
    labels = get_param_list()[:ndim]
    try:
        truths = [result['config'][prior_name] for prior_name in get_prior_list()[:ndim]]
    except KeyError:
        truths = []
        for prior_name in get_prior_list()[:ndim]:
            if prior_name != "rho_0":
                truths.append(result['config'][prior_name])
            else:
                truths.append(result['config']['density'])
    fig = corner.corner(
        result['samples'],
        weights=result['weights'],
        labels=labels,
        range=[0.99999, 0.99999, 0.99999, 0.99999, 0.99999][:ndim],
        truths=truths,
        show_titles=True)
    fig.axes[1].set_title(f"Fit title", loc='left')
    fig.axes[1].text(0, 1, info, verticalalignment='top')
    if save:
        plt.savefig(f"{save}corner.png", dpi=200)
    # plt.show()


def solve_multinest(LogLikelihood, Prior, n_dims, **kwargs):
    from pymultinest.solve import run, Analyzer, solve
    kwargs['n_dims'] = n_dims
    files_temporary = False
    if 'outputfiles_basename' not in kwargs:
        files_temporary = True
        tempdir = tempfile.mkdtemp('pymultinest')
        kwargs['outputfiles_basename'] = tempdir + '/'
    outputfiles_basename = kwargs['outputfiles_basename']

    def SafePrior(cube, ndim, nparams):
        a = np.array([cube[i] for i in range(n_dims)])
        b = Prior(a)
        for i in range(n_dims):
            cube[i] = b[i]


    def SafeLoglikelihood(cube, ndim, nparams, lnew):
        a = np.array([cube[i] for i in range(n_dims)])
        l = float(LogLikelihood(a))
        if not np.isfinite(l):
            warn('WARNING: loglikelihood not finite: %f\n' % (l))
            warn('         for parameters: %s\n' % a)
            warn('         returned very low value instead\n')
            return -1e100
        return l


    kwargs['LogLikelihood'] = SafeLoglikelihood
    kwargs['Prior'] = SafePrior
    run(**kwargs)

    analyzer = Analyzer(n_dims, outputfiles_basename=outputfiles_basename)
    stats = analyzer.get_stats()
    samples = analyzer.get_equal_weighted_posterior()[:, :-1]

    # if files_temporary:
    #     shutil.rmtree(tempdir, ignore_errors=True)

    return dict(logZ=stats['nested sampling global log-evidence'],
                logZerr=stats['nested sampling global log-evidence error'],
                samples=samples,
                )