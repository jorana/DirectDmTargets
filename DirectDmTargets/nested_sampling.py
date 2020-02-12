"""Do a likelihood fit. The class NestedSamplerStatModel is used for fitting applying the bayesian alogorithm nestle"""

from datetime import datetime
import json
import multiprocessing
import os
from scipy import special as spsp
import corner
import matplotlib.pyplot as plt
import nestle
import shutil

from .halo import *
from .statistics import *
from .utils import *
from pymultinest.solve import run, Analyzer, solve
from .context import *

def default_nested_save_dir():
    return 'nested'


class NestedSamplerStatModel(StatModel):
    known_parameters = get_param_list()

    def __init__(self, *args):
        StatModel.__init__(self, *args)
        self.tol = 0.1
        self.nlive = 1024
        self.sampler = 'nestle'
        self.log = {'did_run': False, 'saved_in': None, 'tmp_dir':None, 'garbage_bin': []}
        self.config['start'] = datetime.now()  # .date().isoformat()
        self.config['notes'] = "default"
        self.result = False
        # self.save_dir = False
        self.set_fit_parameters(['log_mass', 'log_cross_section'])
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tVERBOSE ENABLED')
            if self.verbose > 1:
                print(f"NestedSamplerStatModel::\t{now(self.config['start'])}\n\tSUPERVERBOSE ENABLED\n\tyou want to "
                      f"know it all? Here we go sit back and be blown by my output!")

    def set_fit_parameters(self, params):
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tsetting fit parameters to {params}')
        if not type(params) == list:
            raise TypeError("Set the parameter names in a list of strings")
        for param in params:
            if param not in self.known_parameters:
                err_message = f"{param} does not match any of the known parameters try any of {self.known_parameters}"
                raise NotImplementedError(err_message)
        if not params == self.known_parameters[:len(params)]:
            err_message = f"The parameters are not input in the correct order. Please" \
                          f" insert {self.known_parameters[:len(params)]} rather than {params}."
            raise NameError(err_message)
        self.fit_parameters = params

    def check_did_run(self):
        if not self.log['did_run']:
            if self.verbose:
                print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tdid not run yet, lets fire it up!')
            if self.sampler == 'nestle':
                self.run_nestle()
            elif self.sampler == 'multinest':
                self.run_multinest()
            else:
                raise NotImplementedError(f'No such sampler as {self.sampler}, perhaps try nestle or multinest')
        elif self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tdid run')

    def check_did_save(self):
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tdid not save yet, we dont want to lose '
                  f'our results so better do it now')
        if self.log['saved_in'] is None:
            self.save_results()

    def log_probability_nested(self, parameter_vals, parameter_names):
        """

        :param parameter_vals: the values of the model/benchmark considered as the truth
        # :param parameter_values: the values of the parameters that are being varied
        :param parameter_names: the names of the parameter_values
        :return:
        """
        # if self.verbose:
        #     print(f"'"NestedSamplerStatModel::\t{now(self.config["start"])}\n\t
        if self.verbose > 1:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tSUPERVERBOSE\tthere we go! Find that log probability')
        evaluated_rate = self.eval_spectrum(parameter_vals, parameter_names)[
            'counts']

        ll = log_likelihood(self.benchmark_values, evaluated_rate)
        if np.isnan(ll):
            raise ValueError(
                f"Returned NaN from likelihood. ll = {ll}")
        if self.verbose > 1:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tSUPERVERBOSE\tfound it! returning the log likelihood')
        return ll

    def log_prior_transform_nested(self, x, x_name):
        if self.verbose > 1:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tSUPERVERBOSE\tdoing some transformations for nestle/multinest '
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
        if self.verbose > 1:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tSUPERVERBOSE\tdoing _log_probability_nested'
                  f'\n\t\tooph, what a nasty function to do some transformations behind the scenes')
        ndim = len(theta)
        return self.log_probability_nested(theta, self.known_parameters[:ndim])

    def _log_prior_transform_nested(self, theta):
        if self.verbose > 1:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tSUPERVERBOSE\tdoing _log_prior_transform_nested'
                  f'\n\t\tooph, what a nasty function to do some transformations behind the scenes')
        result = [self.log_prior_transform_nested(val, self.known_parameters[i]) for i, val in enumerate(theta)]
        return np.array(result)

    def run_nestle(self):
        assert self.sampler == 'nestle', f'Trying to run nestle but initialization requires {self.sampler}'
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tWe made it to my core function, lets do that optimization')
        method = 'multi'  # use MutliNest algorithm
        ndim = len(self.fit_parameters)
        tol = self.tol  # the stopping criterion
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\there we go! We are going to fit:\n\t{ndim} parameters\n')
        assert_str = f"Unknown configuration of fit pars: {self.fit_parameters}"
        assert self.fit_parameters == self.known_parameters[:ndim], assert_str
        try:
            print(f'run_nestle::\t{now(self.config["start"])}\n\tstart_fit for %i parameters' % ndim)
            if self.verbose:
                print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tbeyond this point, there is nothing '
                      f"I can say, you'll have to wait for my lower level "
                      f'algorithms to give you info, see you soon!')
            start = datetime.now()
            self.result = nestle.sample(self._log_probability_nested,
                                        self._log_prior_transform_nested,
                                        ndim,
                                        method=method,
                                        npoints=self.nlive,
                                        dlogz=tol)
            end = datetime.now(self.config['start'])
            dt = end - start
            print(f'run_nestle::\t{now(self.config["start"])}\n\tfit_done in %i s (%.1f h)' % (dt.seconds, dt.seconds / 3600.))
            if self.verbose > 1:
                print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tSUPERVERBOSE\tWe are back!')
        except ValueError as e:
            print(f'Nestle did not finish due to a ValueError. Was running with'
                  f'\n{len(self.fit_parameters)} for fit ''parameters ' f'{self.fit_parameters}')
            raise e
        self.log['did_run'] = True
        try:
            self.config['fit_time'] = dt.seconds
        except NameError:
            self.config['fit_time'] = -1
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tFinished with running optimizer!')

    # def self.SafePrior(self.cube, ndim, nparams):
    #     return self._log_probability_nested(cube[:n_dims])
    #
    # def SafeLoglikelihood(cube, ndim, nparams, lnew):
    #     print(f'SafeLoglikelihood called with:\ncube{cube},\ndim{ndim},\n nparams {nparams}, lnew\t{lnew})')
    #     return

    def run_multinest(self):
        assert self.sampler == 'multinest', f'Trying to run multinest but initialization requires {self.sampler}'
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tWe made it to my core function, lets do that optimization')
        # method = 'multi'  # use MutliNest algorithm
        n_dims = len(self.fit_parameters)
        tol = self.tol  # the stopping criterion
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\there we go! We are going to fit:\n\t{n_dims} parameters\n')
        save_at = self.get_save_dir()
        assert_str = f'Unknown configuration of fit pars: {self.fit_parameters}'
        assert self.fit_parameters == self.known_parameters[:n_dims], assert_str
        try:
            print(f'run_multinest::\t{now(self.config["start"])}\n\tstart_fit for %i parameters' % n_dims)
            if self.verbose:
                print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tbeyond this point, there is nothing '
                      f"I can say, you'll have to wait for my lower level "
                      f'algorithms to give you info, see you soon!')
            start = datetime.now()

            # Multinest saves output to a folder. First write to the tmp folder, move it to the results folder later
            tmp_folder = self.get_tmp_dir()
            # save_at_temp = f'{tmp_folder}multinest_{os.getpid()}'
            save_at_temp = f'{tmp_folder}multinest'
            # Need try except for making sure the tmp folder is removed
            # try:
            # copied from https://github.com/JohannesBuchner/PyMultiNest/blob/master/pymultinest/solve.py
            # def SafePrior(cube, ndim, nparams):
            #     return self._log_probability_nested(cube[:n_dims])
            #     try:
            #         a = np.array([cube[i] for i in range(n_dims)])
            #         b = self._log_prior_transform_nested(a)
            #         for i in range(n_dims):
            #             cube[i] = b[i]
            #     except Exception as e:
            #         import sys
            #         sys.stderr.write('ERROR in prior: %s\n' % e)
            #         sys.exit(1)
            #
            # def SafeLoglikelihood(cube, ndim, nparams, lnew):
            #     print(f'SafeLoglikelihood called with:\ncube{cube},\ndim{ndim},\n nparams {nparams}, lnew\t{lnew})')
            #     return self._log_probability_nested(cube[:n_dims])
            #     try:
            #         a = np.array([cube[i] for i in range(n_dims)])
            #         # l = float(self._log_probability_nested(a))
            #         l = self._log_probability_nested(a)
            #         if not np.isfinite(l):
            #             import sys
            #             sys.stderr.write('WARNING: loglikelihood not finite: %f\n' % (l))
            #             sys.stderr.write('         for parameters: %s\n' % a)
            #             sys.stderr.write('         returned very low value instead\n')
            #             return -1e100
            #         return l
            #     except Exception as e:
            #         import sys
            #         sys.stderr.write('ERROR in loglikelihood: %s\n' % e)
            #         sys.exit(1)

            solve(
                LogLikelihood=self._log_probability_nested,#SafeLoglikelihood,
                Prior=self._log_prior_transform_nested, #SafePrior,
                n_live_points=self.nlive,
                n_dims=n_dims,
                outputfiles_basename=save_at_temp,
                verbose=True,
                evidence_tolerance=tol
                )
            self.result = save_at_temp
            # except OSError:
            # #     # Multinest in multiprocessing does run into an OS error since it opens multiple threads that are later
            # #     # recombined. These threads that are opened complain that they cannot find some file since their result
            # #     #  is added to first of their results.
            #     print('Ran into a save OSError')
            # except IOError:
            #     print('Ran into a save OSError')
#             except:
#                 print(f'run_multinest::\tFAILED. Remove tmp folder!')
#                 shutil.rmtree(tmp_folder)

            # Open a save-folder after succesful running multinest. Move the multinest results there.
            # save_at = self.get_save_dir()
            check_folder_for_file(save_at)
            assert tmp_folder[-1] == '/', 'make sure that tmp_folder ends at '/''
            copy_multinest = save_at + tmp_folder.split('/')[-2]
            print(f'copy {tmp_folder} to {copy_multinest}')
#             if not os.path.exists(copy_multinest) and os.path.exists(tmp_folder):
#                 try:
#                     shutil.copytree(tmp_folder, copy_multinest)
#                 except FileExistsError:
#                     pass
#                 except shutil.Error:
#                     pass
            self.log['garbage_bin'].append(tmp_folder)
            # if not os.path.exists(save_at):
            #     shutil.copytree(tmp_folder, save_at)
            # else:
            #     shutil.copy2(tmp_folder, save_at)
            # assert not os.path.exists(tmp_folder), f'the tmp folder {tmp_folder} was not moved correctly to {save_at}'

            end = datetime.now()
            dt = end - start
            print(f'run_multinest::\t{now(self.config["start"])}\n\tfit_done in %i s (%.1f h)' % (dt.seconds, dt.seconds / 3600.))
            if self.verbose > 1:
                print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tSUPERVERBOSE\tWe are back!')
        except ValueError as e:
            print(f'Multinest did not finish due to a ValueError. Was running with'
                  f'\n{len(self.fit_parameters)} for fit parameters {self.fit_parameters}')
            raise e
        self.log['did_run'] = True
        try:
            self.config['fit_time'] = dt.seconds
        except NameError:
            self.config['fit_time'] = -1
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tFinished with running Multinest!')

    def empty_garbage(self):
        for file in self.log['garbage_bin']:
            print(f'delete {file}')
            if os.path.exists(file):
                try:
                    shutil.rmtree(file)
                except FileNotFoundError:
                    pass
            else:
                print(f'Could not find {file} that is in the garbage bin?')

    def get_summary(self):
        if self.verbose:
            pass
        print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tgetting the summary (or at'
              f"least trying) let's first see if I did run")
        self.check_did_run()
        if self.verbose:
            print(f"NestedSamplerStatModel::\t{now(self.config['start'])}\n\tAlright, that's done. Let's get some "
                  f"info. I'm not ging to print too much here")
        # keep a dictionary of all the results
        resdict = {}

        if self.sampler == 'multinest':
            print('start analyzer of results')
            analyzer = Analyzer(len(self.fit_parameters), outputfiles_basename=self.result)
            # Taken from multinest.solve
	    self.result = analyzer.get_stats()
            samples = analyzer.get_equal_weighted_posterior()[:, :-1]

            print('parameter values:')
            for name, col in zip(self.fit_parameters, samples.transpose()):
                print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))
                resdict[name + '_fit_res'] = ('{0:5.2f} +/- {1:5.2f}'.format(col.mean(),col.std()))
                if 'log_' in name:
                    resdict[name[4:] + '_fit_res'] = '%.3g +/- %.2g' % (10 ** col.mean(), 10 ** (col.mean()) * np.log(10) * col.std())
                    print('\t', name[4:], resdict[name[4:] + '_fit_res'])
            resdict['n_samples'] = len(samples.transpose()[0])
            # Pass the samples to the self.result to be saved.
	    self.result['samples'] = samples
        elif self.sampler == 'nestle':
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
                print('\t', key, resdict[key + '_fit_res'])
                if 'log_' in key:
                    resdict[key[4:] + '_fit_res'] = '%.3g +/- %.2g' % (
                        10 ** p[i], 10 ** (p[i]) * np.log(10) * np.sqrt(cov[i, i]))
                    print('\t', key[4:], resdict[key[4:] + '_fit_res'])
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tAlright we got all the info we need, '
                  f"let's return it to whomever asked for it")
        return resdict

    def get_save_dir(self, force_index = False, hash = None):
        print()
        if (not self.log['saved_in']) or force_index:
            self.log['saved_in'] = open_save_dir(f'{default_nested_save_dir()}_{self.sampler}', force_index=force_index, hash=hash)
        if self.verbose:
            print(f'NestedSamplerStatModel::\tget_save_dir\tsave_dir = {self.log["saved_in"]}')
        return self.log['saved_in']

    def get_tmp_dir(self, force_index = False, hash = None):
        if (not self.log['tmp_dir']) or force_index:
            self.log['tmp_dir'] = open_save_dir(f'{self.sampler}', base=context['tmp_folder'], force_index=force_index, hash=hash)
        if self.verbose:
            print(f'NestedSamplerStatModel::\tget_tmp_dir\ttmp_dir = {self.log["tmp_dir"]}')
        return self.log['tmp_dir']

    def save_results(self, force_index=False):
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tAlmost there! We are about to save the '
                  f'results. But first do some checks, did we actually run?')
        # save fit parameters to config
        self.config['fit_parameters'] = self.fit_parameters
        self.check_did_run()
        save_dir = self.get_save_dir(force_index=force_index)
        fit_summary = self.get_summary()
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tAlright all set, let put all that info'
                  f' in {save_dir} and be done with it')
        # save the config, chain and flattened chain
        if 'HASH' in save_dir or os.path.exists(save_dir+'config.json'):
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
        print(f'save_results::\t{now(self.config["start"])}\n\tdone_saving')

    def show_corner(self, save=True):
        if self.verbose:
            print(f"NestedSamplerStatModel::\t{now(self.config['start'])}\n\tLet's do some graphics, I'll make you a "
                  f"nice corner plot just now")
        self.check_did_save()
        save_dir = self.log['saved_in']
        combined_results = load_nestle_samples_from_file(save_dir)
        nestle_corner(combined_results, save_dir)
        if self.verbose:
            print(f'NestedSamplerStatModel::\t{now(self.config["start"])}\n\tEnjoy the plot. Maybe you do want to'
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
    print(f'load_nestle_samples::\t{now()}\n\tloading', load_dir)
    keys = ['config', 'res_dict', 'h', 'logl', 'logvol', 'logz', 'logzerr',
            'ncall', 'niter', 'samples', 'weights']
    result = {}
    for key in keys:
        result[key] = np.load(load_dir + key + '.npy', allow_pickle=True)
        if key == 'config' or key == 'res_dict':
            result[key] = result[key].item()
    print(f"load_nestle_samples::\t{now()}\n\tdone loading\naccess result with:\n{keys}")
    return result

def load_multinest_samples_from_file(load_dir):
    keys = ['config', 'res_dict', 'wighted_samples']
    result = {}
    for key in keys:
        result[key] = np.load(load_dir + key + '.npy', allow_pickle=True)
        if key == 'config' or key == 'res_dict':
            result[key] = result[key].item()
    print(f"load_multinest_samples_from_file::\t{now()}\n\tdone loading\naccess result with:\n{keys}")
    return result


def nestle_corner(result, save=False):
    info = "$M_\chi}$=%.2f" % 10 ** np.float(result['config']['mw'])
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
    plt.show()
