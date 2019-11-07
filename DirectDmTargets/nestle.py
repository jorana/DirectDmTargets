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

#TODO want to make this a class dependent on MCMCStatModel ?
class NestleStatModel(StatModel):
    known_parameters = ['log_mass',
                        'log_cross_section',
                        'v_0',
                        'v_esc',
                        'density']
    def __init__(self, *args):
        StatModel.__init__(self, *args)
        self.tol = 0.1
        self.nlive = 1024
        self.fit_parameters = ['log_mass', 'log_cross_section']
        self.log = {'did_run': False}
        self.config['start'] = datetime.datetime.now()  # .date().isoformat()
        self.config['notes'] = "default"
        self.result = False

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

    def log_probability_nestle(self, parameter_vals, parameter_names):
        """

        :param parameter_vals: the values of the model/benchmark considered as
        the truth
        # :param parameter_values: the values of the parameters that are being
        varied
        :param parameter_names: the names of the parameter_values
        :return:
        """

        model = self.eval_spectrum(parameter_vals, parameter_names)

        ll = log_likelihood(model, self.benchmark_values)
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
            res = m + s * spsp.ndtri(x)
            if a < res < b:
                return res
            else:
                return -np.inf
        else:
            raise TypeError(
                f"unknown prior type '"
                f"{self.config['prior'][x_name]['prior_type']}', choose either "
                f"gauss or flat")

    # TODO build in reduncancy for incorrect order of parameters
    def _log_probability_nestle(self, theta):
        ndim = len(self.fit_parameters)
        return self.log_probability_nestle(theta, self.known_parameters[:ndim])

    def _log_prior_transform_nestle(self, theta):
        ndim = len(self.fit_parameters)
        result = [self.log_prior_transform_nestle(val, self.known_parameters[i])
                  for i, val in enumerate(theta)]
        return np.array(result)

    def run_nestle(self):
        method = 'multi'  # use MutliNest algorithm
        ndim = len(self.fit_parameters)
        tol = self.tol  # the stopping criterion
        try:
            # todo
            print("run_nestle::\tstart_fit for %i parameters"%ndim)
            start = datetime.datetime.now()
            self.result = nestle.sample(self._log_probability_nestle,
                                        self._log_prior_transform_nestle,
                                        ndim,
                                        method=method,
                                        npoints=self.nlive,
                                        dlogz=tol)
            end = datetime.datetime.now()
            dt = end - start
            print("run_nestle::\tfit_done in %i s"%(dt.seconds))
        except ValueError as e:
            print(
                f"Nestle did not finish due to a ValueError. Was running with\n"
                f"pos={self.pos.shape} nsteps = {self.nsteps}, walkers = "
                f"{self.nwalkers}, ndim = "
                f"{len(self.fit_parameters)} for fit parameters "
                f"{self.fit_parameters}")
            raise e
        self.log['did_run'] = True
        try:
            self.config['fit_time'] = dt.seconds
        except NameError:
            self.config['fit_time'] = -1

    def get_summary(self):
        self.check_did_run()
        logZnestle = self.result.logz  # value of logZ
        infogainnestle = self.result.h  # value of the information gain in nats
        logZerrnestle = np.sqrt(infogainnestle / self.nlive)  # estimate of the statistcal uncertainty on logZ

        # re-scale weights to have a maximum of one
        nweights = self.result.weights / np.max(self.result.weights)

        # get the probability of keeping a sample from the weights
        keepidx = np.where(np.random.rand(len(nweights)) < nweights)[0]

        # get the posterior samples
        samples_nestle = self.result.samples[keepidx, :]
        resdict = {}
        # resdict['mnestle_mu'] = np.mean(samples_nestle[:, 0])  # mean of m samples
        # resdict['mnestle_sig'] = np.std(samples_nestle[:, 0])  # standard deviation of m samples
        # resdict['cnestle_mu'] = np.mean(samples_nestle[:, 1])  # mean of c samples
        # resdict['cnestle_sig'] = np.std(samples_nestle[:, 1])  # standard deviation of c samples
        # resdict['ccnestle'] = np.corrcoef(samples_nestle.T)[0, 1]  # correlation coefficient between parameters
        resdict['nestle_nposterior'] = len(samples_nestle)  # number of posterior samples
        resdict['nestle_time'] = self.config['fit_time']  # run time
        resdict['nestle_logZ'] = logZnestle  # log marginalised likelihood
        resdict['nestle_logZerr'] = logZerrnestle  # uncertainty on log(Z)
        resdict['summary'] = self.result.summary()
        resdict['N_posterior_samples '] = len(samples_nestle)
        p, cov = nestle.mean_and_cov(self.result.samples, self.result.weights)
        for i, key in enumerate(self.fit_parameters):
            resdict[key + "_fit_res"] = "{0:5.2f} +/- {1:5.2f}".format(p[i], np.sqrt(cov[i, i]))
            print('\t', key, resdict[key + "_fit_res"])
        return resdict

    def save_results(self, force_index=False):
        self.check_did_run()
        base = 'results/'
        save = 'test_nestle'
        files = os.listdir(base)
        files = [f for f in files if save in f]
        if not save + '0' in files and not force_index:
#             os.makedirs(base + save + '0')
            index = 0
        elif force_index is False:
            index = max([int(f.split(save)[-1]) for f in files]) + 1
        else:
            index = force_index

        save_dir = base + save + str(index) + '/'
        print('save_results::\tusing ' + save_dir)
        if force_index is False:
            os.mkdir(save_dir)
        else:
            assert os.path.exists(save_dir), "specify existing directory, exit"
            for file in os.listdir(save_dir):
                print('save_results::\tremoving ' + save_dir + file)
                os.remove(save_dir + file)
        # save the config, chain and flattened chain
        with open(save_dir + 'config.json', 'w') as file:
            json.dump(convert_config_to_savable(self.config), file, indent=4)

        with open(save_dir + 'res_dict.json', 'w') as file:
            json.dump(convert_config_to_savable(self.get_summary()), file, indent=4)
        np.save(save_dir + 'config.npy',
                convert_config_to_savable(self.config))
        np.save(save_dir + 'res_dict.npy',
                convert_config_to_savable(self.get_summary()))
        for col in self.result.keys():
            if col == 'samples' or type(col) is not dict:
                np.save(save_dir + col + '.npy', self.result[col])
            else:
                np.save(save_dir + col + '.npy',
                        convert_config_to_savable(self.result[col]))
        print("save_results::\tdone_saving")


def is_savable_type(item):
    if type(item) in [list, np.array, np.ndarray, int, str, np.int, np.float,
                      bool, np.float64]:
        return True
    return False


def convert_config_to_savable(config):
    result = config.copy()
    for key in result.keys():
        if is_savable_type(result[key]):
            pass
        elif type(result[key]) == dict:
            result[key] = convert_config_to_savable(result[key])
        else:
            result[key] = str(result[key])
    return result

def load_nestle_samples(item='latest'):
    base = 'results/'
    save = 'test_nestle'
    files = os.listdir(base)
    if item is 'latest':
        item = max([int(f.split(save)[-1]) for f in files if save in f])
    result = {}
    load_dir = base + save + str(item) + '/'
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Cannot find {load_dir} specified by arg: "
                                f"{item}")
    print("loading", load_dir)

    keys = ['config', 'res_dict', 'h', 'logl', 'logvol', 'logz', 'logzerr',
            'ncall', 'niter', 'samples', 'weights']
    for key in keys:
        result[key] = np.load(load_dir + key + '.npy', allow_pickle=True)
        if key is 'config' or key is 'res_dict':
            result[key] = result[key].item()
    print(f"load_nestle_samples::\tdone loading\naccess result with:\n{keys}")
    return result