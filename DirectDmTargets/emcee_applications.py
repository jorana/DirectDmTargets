"""Do a likelihood fit. The class MCMCStatModel is used for fitting applying
the MCMC alogorithm emcee.

MCMC is:
    slower than the nestle package; and
    harder to use since one has to choose the 'right' initial parameters

Nevertheless, the walkers give great insight in how the likelihood-function is
felt by the steps that the walkers make"""

from datetime import datetime
import json
import multiprocessing
import os

import corner
import emcee
import matplotlib.pyplot as plt

from .statistics import *
from .utils import *


def default_emcee_save_dir():
    return 'emcee'


class MCMCStatModel(StatModel):
    known_parameters = ['log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density']

    def __init__(self, *args):
        StatModel.__init__(self, *args)
        self.nwalkers = 50
        self.nsteps = 100
        self.fit_parameters = ['log_mass', 'log_cross_section']
        self.sampler = None
        self.pos = None
        self.log = {'sampler': False, 'did_run': False, 'pos': False}
        self.remove_frac = 0.2
        self.thin = 15
        self.config['start'] = datetime.now()  # .date().isoformat()
        self.config['notes'] = "default"

    def set_fit_parameters(self, params):
        if not type(params) == list:
            raise TypeError("Set the parameter names in a list of strings")
        for param in params:
            if param not in self.known_parameters:
                raise NotImplementedError(f"{param} does not match any of the "
                                          f"known parameters try any of "
                                          f"{self.known_parameters}")
        if not params == self.known_parameters[:len(params)]:
            err_message = f"The parameters are not input in the correct order. " \
                          f"Please insert {self.known_parameters[:len(params)]} rather than {params}."
            raise NameError(err_message)
        self.fit_parameters = params

    def set_pos_full_prior(self, use_pos=None):
        self.log['pos'] = True
        if use_pos is not None:
            self.pos = use_pos
            return
        pos = np.hstack([
            [[np.clip(self.config['prior'][param]['dist'](
                self.config['prior'][param]['param']),
                1.25 * self.config['prior'][param]['range'][0],
                0.75 * self.config['prior'][param]['range'][-1]
            ) for i in range(self.nwalkers)]
                for param in self.fit_parameters]
        ])
        return pos.T

    def set_pos(self, use_pos=None):
        self.log['pos'] = True
        if use_pos is not None:
            print("using specified start position")
            self.pos = use_pos
            return
        nparameters = len(self.fit_parameters)
        keys = get_prior_list()[:nparameters]

        ranges = [self.config['prior'][self.fit_parameters[i]]['range']
                  for i in range(nparameters)]
        pos = []

        for i, key in enumerate(keys):
            val = self.config.get(key)
            a, b = ranges[i]
            if key in []:
                start_at = np.random.uniform(a, b, (self.nwalkers, 1))
            else:
                start_at = val + 0.005 * val * np.random.randn(self.nwalkers, 1)
            start_at = np.clip(start_at, a, b)
            pos.append(start_at)
        pos = np.hstack(pos)
        self.pos = pos

    def set_sampler(self, mult=True):
        ndim = len(self.fit_parameters)
        kwargs = {"threads": multiprocessing.cpu_count()} if mult else {}
        self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim,
                                             self.log_probability,
                                             args=([self.fit_parameters]),
                                             **kwargs)
        self.log['sampler'] = True

    def run_emcee(self):
        if not self.log['sampler']:
            self.set_sampler()
        if not self.log['pos']:
            self.set_pos()
        try:
            start = datetime.now()
            self.sampler.run_mcmc(self.pos, self.nsteps, progress=False)
            end = datetime.now()
        except ValueError as e:
            print(f"MCMC did not finish due to a ValueError. Was running with\n"
                  f"pos={self.pos.shape} nsteps = {self.nsteps}, walkers = "
                  f"{self.nwalkers}, ndim = "
                  f"{len(self.fit_parameters)} for fit parameters "
                  f"{self.fit_parameters}")
            raise e
        self.log['did_run'] = True
        try:
            dt = end - start
            print("run_emcee::\tfit_done in %i s (%.1f h)" % (
                dt.seconds, dt.seconds / 3600.))
            self.config['fit_time'] = dt.seconds
        except NameError:
            self.config['fit_time'] = -1

    def show_walkers(self):
        if not self.log['did_run']:
            self.run_emcee()
        labels = self.fit_parameters
        fig, axes = plt.subplots(len(labels), figsize=(10, 7), sharex=True)
        samples = self.sampler.get_chain()
        for i in range(len(labels)):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)
        axes[-1].set_xlabel("step number")

    def show_corner(self):
        if not self.log['did_run']:
            self.run_emcee()
        print(f"Removing a fraction of {self.remove_frac} of the samples, total"
              f"number of removed samples = {self.nsteps * self.remove_frac}")
        flat_samples = self.sampler.get_chain(
            discard=int(self.nsteps * self.remove_frac),
            thin=self.thin,
            flat=True
        )
        truths = [self.config[prior_name] for prior_name in
                  get_prior_list()[:len(self.fit_parameters)]]

        corner.corner(flat_samples, labels=self.fit_parameters, truths=truths)

    def save_results(self, force_index=False):
        # save fit parameters to config
        self.config['fit_parameters'] = self.fit_parameters
        if not self.log['did_run']:
            self.run_emcee()
        # open a folder where to save to results
        save_dir = open_save_dir(default_emcee_save_dir(), force_index)
        # save the config, chain and flattened chain
        with open(save_dir + 'config.json', 'w') as fp:
            json.dump(convert_dic_to_savable(self.config), fp, indent=4)
        np.save(save_dir + 'config.npy',
                convert_dic_to_savable(self.config))
        np.save(save_dir + 'full_chain.npy', self.sampler.get_chain())
        np.save(save_dir + 'flat_chain.npy', self.sampler.get_chain(
            discard=int(self.nsteps * self.remove_frac), thin=self.thin,
            flat=True))
        print("save_results::\tdone_saving")


def load_chain_emcee(load_from=default_emcee_save_dir(), item='latest'):
    base = get_result_folder()
    save = load_from
    files = os.listdir(base)
    if item is 'latest':
        item = max([int(f.split(save)[-1]) for f in files if save in f])
    result = {}
    load_dir = base + save + str(item) + '/'
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Cannot find {load_dir} specified by arg: "
                                f"{item}")
    print("loading", load_dir)

    keys = ['config', 'full_chain', 'flat_chain']

    for key in keys:
        result[key] = np.load(load_dir + key + '.npy', allow_pickle=True)
        if key == 'config':
            result[key] = result[key].item()
    print(f"done loading\naccess result with:\n{keys}")
    return result


def emcee_plots(result, save=False, plot_walkers=True):
    if not type(save) is bool:
        assert os.path.exists(save), f"invalid path '{save}'"
        if not save[-1] == "/":
            save += "/"
    info = "$M_\chi}$=%.2f" % 10 ** np.float(result['config']['mw'])
    for prior_key in result['config']['prior'].keys():
        try:
            mean = result['config']['prior'][prior_key]['mean']
            info += f"\n{prior_key} = {mean}"
        except KeyError:
            pass
    nsteps, nwalkers, ndim = np.shape(result['full_chain'])

    for str_inf in ['notes', 'start', 'fit_time', 'poisson', 'nwalkers', 'nsteps',
                    'n_energy_bins']:
        try:
            info += f"\n{str_inf} = %s" % result['config'][str_inf]
            if str_inf is 'start':
                info = info[:-7]
            if str_inf == 'fit_time':
                info += 's (%.1f h)' % (result['config'][str_inf] / 3600.)

        except KeyError:
            pass
    info += "\nnwalkers = %s" % nwalkers
    info += "\nnsteps = %s" % nsteps
    labels = get_param_list()[:ndim]
    truths = [result['config'][prior_name] for prior_name in
              get_prior_list()[:ndim]]
    fig = corner.corner(
        result['flat_chain'],
        labels=labels,
        range=[0.99999, 0.99999, 0.99999, 0.99999, 0.99999][:ndim],
        truths=truths,
        show_titles=True)
    fig.axes[1].set_title(f"{result['config']['detector']}", loc='left')
    fig.axes[1].text(0, 1, info, verticalalignment='top')
    if save:
        plt.savefig(f"{save}corner.png", dpi=200)
    plt.show()

    if plot_walkers:
        fig, axes = plt.subplots(len(labels), figsize=(10, 5), sharex=True)
        for i in range(len(labels)):
            ax = axes[i]
            ax.plot(result['full_chain'][:, :, i], "k", alpha=0.3)
            ax.axhline(truths[i])
            ax.set_xlim(0, len(result['full_chain']))
            ax.set_ylabel(labels[i])
            ax.yaxis.set_label_coords(-0.1, 0.5)

        axes[-1].set_xlabel("step number")
        if save:
            plt.savefig(f"{save}flat_chain.png", dpi=200)
        plt.show()
