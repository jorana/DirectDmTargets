"""Do a likelihood fit. The class NestleStatModel is used for fitting applying
the baseyan alogorithm nestle"""

import datetime
import json
import multiprocessing
import os
from scipy import special as spsp
import corner
import emcee
import matplotlib.pyplot as plt
import nestle

from .halo import *
from .statistics import *
from .utils import *


def default_nestle_save_dir():
    return 'nestle'


class NestleStatModel(StatModel):
    known_parameters = get_param_list()

    def __init__(self, *args):
        StatModel.__init__(self, *args)
        self.tol = 0.1
        self.nlive = 1024
        self.log = {'did_run': False, 'saved_in': None}
        self.config['start'] = datetime.datetime.now()  # .date().isoformat()
        self.config['notes'] = "default"
        self.result = False
        self.set_fit_parameters(['log_mass', 'log_cross_section'])

    def set_fit_parameters(self, params):
        if not type(params) == list:
            raise TypeError("Set the parameter names in a list of strings")
        for param in params:
            if param not in self.known_parameters:
                raise NotImplementedError(f"{param} does not match any of the "
                                          f"known parameters try any of "
                                          f"{self.known_parameters}")
        if not params == self.known_parameters[:len(params)]:
            raise NameError(f"The parameters are not input in the correct order"
                            f". Please insert "
                            f"{self.known_parameters[:len(params)]} rather than"
                            f" {params}.")
        self.fit_parameters = params

    def check_did_run(self):
        if not self.log['did_run']:
            self.run_nestle()

    def check_did_save(self):
        if self.log['saved_in'] is None:
            self.save_results()

    def log_probability_nestle(self, parameter_vals, parameter_names):
        """

        :param parameter_vals: the values of the model/benchmark considered as
        the truth
        # :param parameter_values: the values of the parameters that are being
        varied
        :param parameter_names: the names of the parameter_values
        :return:
        """
        evaluated_rate = self.eval_spectrum(parameter_vals, parameter_names)[
            'counts']

        ll = log_likelihood(self.benchmark_values, evaluated_rate)
        if np.isnan(ll):
            raise ValueError(
                f"Returned NaN from likelihood. ll = {ll}")
        return ll

    def log_prior_transform_nestle(self, x, x_name):
        if self.config['prior'][x_name]['prior_type'] == 'flat':
            a, b = self.config['prior'][x_name]['param']
            return x * (b - a) + a
        elif self.config['prior'][x_name]['prior_type'] == 'gauss':
            a, b = self.config['prior'][x_name]['range']
            m, s = self.config['prior'][x_name]['param']
            aprime = spsp.ndtr((a - m) / s)
            bprime = spsp.ndtr((b - m) / s)
            xprime = x * (bprime - aprime) + aprime
            res = m + s * spsp.ndtri(xprime)
            return res
        else:
            raise TypeError(
                f"unknown prior type '"
                f"{self.config['prior'][x_name]['prior_type']}', choose either "
                f"gauss or flat")

    def _log_probability_nestle(self, theta):
        ndim = len(theta)
        return self.log_probability_nestle(theta, self.known_parameters[:ndim])

    def _log_prior_transform_nestle(self, theta):
        result = [self.log_prior_transform_nestle(val, self.known_parameters[i])
                  for i, val in enumerate(theta)]
        return np.array(result)

    def run_nestle(self):
        method = 'multi'  # use MutliNest algorithm
        ndim = len(self.fit_parameters)
        tol = self.tol  # the stopping criterion
        assert_str = f"Unknown configuration of fit pars: {self.fit_parameters}"
        assert self.fit_parameters == self.known_parameters[:ndim], assert_str
        try:
            print("run_nestle::\tstart_fit for %i parameters" % ndim)
            start = datetime.datetime.now()
            self.result = nestle.sample(self._log_probability_nestle,
                                        self._log_prior_transform_nestle,
                                        ndim,
                                        method=method,
                                        npoints=self.nlive,
                                        dlogz=tol)
            end = datetime.datetime.now()
            dt = end - start
            print("run_nestle::\tfit_done in %i s (%.1f h)" % (
                dt.seconds, dt.seconds / 3600.))
        except ValueError as e:
            print(
                f"Nestle did not finish due to a ValueError. Was running with\n"
                f"{len(self.fit_parameters)} for fit parameters "
                f"{self.fit_parameters}")
            raise e
        self.log['did_run'] = True
        try:
            self.config['fit_time'] = dt.seconds
        except NameError:
            self.config['fit_time'] = -1

    def get_summary(self):
        # taken from
        # mattpitkin.github.io/samplers-demo/pages/samplers-samplers-everywhere/#Nestle
        self.check_did_run()
        # estimate of the statistical uncertainty on logZ
        logZerrnestle = np.sqrt(self.result.h / self.nlive)
        # re-scale weights to have a maximum of one
        nweights = self.result.weights / np.max(self.result.weights)
        # get the probability of keeping a sample from the weights
        keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]
        # get the posterior samples
        samples_nestle = self.result.samples[keepidx, :]
        resdict = {}
        # estimate of the statistcal uncertainty on logZ
        resdict['nestle_nposterior'] = len(samples_nestle)
        resdict['nestle_time'] = self.config['fit_time']  # run time
        resdict['nestle_logZ'] = self.result.logz  # log marginalised likelihood
        resdict['nestle_logZerr'] = logZerrnestle  # uncertainty on log(Z)
        resdict['summary'] = self.result.summary()
        p, cov = nestle.mean_and_cov(self.result.samples, self.result.weights)
        for i, key in enumerate(self.fit_parameters):
            resdict[key + "_fit_res"] = \
                ("{0:5.2f} +/- {1:5.2f}".format(p[i], np.sqrt(cov[i, i])))
            print('\t', key, resdict[key + "_fit_res"])
            if "log_" in key:
                resdict[key[4:] + "_fit_res"] = "%.3g +/- %.2g" % (
                    10 ** p[i], 10 ** (p[i]) * np.log(10) * np.sqrt(cov[i, i]))
                print('\t', key[4:], resdict[key[4:] + "_fit_res"])
        return resdict

    def save_results(self, force_index=False):
        # save fit parameters to config
        self.config['fit_parameters'] = self.fit_parameters
        self.check_did_run()
        save_dir = open_save_dir(default_nestle_save_dir(), force_index)
        fit_summary = self.get_summary()
        # save the config, chain and flattened chain
        with open(save_dir + 'config.json', 'w') as file:
            json.dump(convert_dic_to_savable(self.config), file, indent=4)
        with open(save_dir + 'res_dict.json', 'w') as file:
            json.dump(convert_dic_to_savable(fit_summary), file, indent=4)
        np.save(save_dir + 'config.npy', convert_dic_to_savable(self.config))
        np.save(save_dir + 'res_dict.npy', convert_dic_to_savable(fit_summary))
        for col in self.result.keys():
            if col == 'samples' or type(col) is not dict:
                np.save(save_dir + col + '.npy', self.result[col])
            else:
                np.save(save_dir + col + '.npy',
                        convert_dic_to_savable(self.result[col]))
        self.log['saved_in'] = save_dir
        print("save_results::\tdone_saving")

    def show_corner(self, save=True):
        self.check_did_save()
        save_dir = self.log['saved_in']
        combined_results = load_nestle_samples_from_file(save_dir)
        nestle_corner(combined_results, save_dir)


def is_savable_type(item):
    if type(item) in [list, np.array, np.ndarray, int, str, np.int, np.float,
                      bool, np.float64]:
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


def load_nestle_samples(load_from=default_nestle_save_dir(), item='latest'):
    base = get_result_folder()
    save = load_from
    files = os.listdir(base)
    if item is 'latest':
        item = max([int(f.split(save)[-1]) for f in files if save in f])

    load_dir = base + save + str(item) + '/'
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Cannot find {load_dir} specified by arg: "
                                f"{item}")
    return load_nestle_samples_from_file(load_dir)


def load_nestle_samples_from_file(load_dir):
    print("load_nestle_samples::\tloading", load_dir)
    keys = ['config', 'res_dict', 'h', 'logl', 'logvol', 'logz', 'logzerr',
            'ncall', 'niter', 'samples', 'weights']
    result = {}
    for key in keys:
        result[key] = np.load(load_dir + key + '.npy', allow_pickle=True)
        if key is 'config' or key is 'res_dict':
            result[key] = result[key].item()
    print(f"load_nestle_samples::\tdone loading\naccess result with:\n{keys}")
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
    for str_inf in ['detector', 'notes', 'start', 'fit_time', 'poisson',
                    'n_energy_bins']:
        try:
            info += f"\n{str_inf} = %s" % result['config'][str_inf]
            if str_inf is 'start':
                info = info[:-7]
            if str_inf == 'fit_time':
                info += 's (%.1f h)' % (result['config'][str_inf] / 3600.)

        except KeyError:
            pass
    labels = get_param_list()[:ndim]
    #TODO remove duplicates like rho_0 and density
    try:
        truths = [result['config'][prior_name] for prior_name in
                  get_prior_list()[:ndim]]
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
