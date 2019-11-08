import datetime
import json
import multiprocessing
import os

import corner
import emcee
import matplotlib.pyplot as plt

from .halo import *
from .statistics import *


class MCMCStatModel(StatModel):
    known_parameters = ['log_mass',
                        'log_cross_section',
                        'v_0',
                        'v_esc',
                        'density']

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
        self.config['start'] = datetime.datetime.now()  # .date().isoformat()
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
            raise NameError(f"The parameters are not input in the correct order"
                            f". Please insert "
                            f"{self.known_parameters[:len(params)]} rather than"
                            f" {params}.")
        self.fit_parameters = params

    def _set_pos(self, use_pos=None):
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
        # for i, p in enumerate(self.fit_parameters):
        #     if 'log' in p:
        #         pos[i] = 10 ** pos[i]
        #     # #TODO workaround
        # if 'cross' in p:
        #     pos[i] = 1e-45 + 1e-45 * np.random.rand(self.nwalkers)
        # if 'mass' in p:
        #     pos[i] = 50 + 50 * np.random.rand(self.nwalkers)
        # self.pos = pos.T
        return pos.T

    def set_pos(self, use_pos=None):
        self.log['pos'] = True
        if use_pos is not None:
            print("using specified start position")
            self.pos = use_pos
            return
        nparameters = len(self.fit_parameters)
        keys = ['mw', 'sigma', 'v_0', 'v_esc', 'rho_0'][:nparameters]
        vals = [self.config.get(key) for key in keys]

        ranges = [self.config['prior'][self.fit_parameters[i]]['range']
                  for i in range(nparameters)]
        # for i, param in enumerate(self.fit_parameters):
        #     if 'log' in param:
        #         ranges[i] = [10 ** this_range for this_range in ranges[i]]
#         pos = np.hstack([
#             np.clip(
#                 #                 val + 0.001 * val * np.random.randn(self.nwalkers, 1),
#                 val + 0.25 * val * np.random.randn(self.nwalkers, 1),
#                 #                 val + 0.5 * val * np.abs(
#                 #                     np.random.randn(self.nwalkers, 1)),
#                 1. * ranges[i][0],
#                 1. * ranges[i][-1])
#             for i, val in enumerate(vals)])
        # Change
        pos = []
        for i, key in enumerate(keys):
            val = self.config.get(key)
            a, b = ranges[i]
            if key in ['v_0', 'v_esc', 'rho_0']:
                start_at = np.random.uniform(a, b, (self.nwalkers, 1))
            else:
                start_at = val + 0.05 * val * np.random.randn(self.nwalkers, 1)
            start_at = np.clip(start_at, a, b)
            pos.append(start_at)
        pos = np.hstack(pos)
        self.pos = pos
    #         self.pos = self._set_pos()

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
            start = datetime.datetime.now()
            self.sampler.run_mcmc(self.pos, self.nsteps, progress=False)
            end = datetime.datetime.now()
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
            print("run_emcee::\tfit_done in %i s (%.1f h)"%(dt.seconds, dt.seconds/3600.))
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
        truths = [self.config['mw'],
                  self.config['sigma'],
                  self.config['v_0'],
                  self.config['v_esc'],
                  self.config['rho_0']
                  ]
        corner.corner(flat_samples, labels=self.fit_parameters,
                            truths=truths[:len(self.fit_parameters)])

    def save_results(self, force_index=False):
        if not self.log['did_run']:
            self.run_emcee()
        base = 'results/'
        save = 'test_emcee'
        files = os.listdir(base)
        files = [f for f in files if save in f]
        if not save + '0' in files and not force_index:
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
        with open(save_dir + 'config.json', 'w') as fp:
            json.dump(convert_config_to_savable(self.config), fp, indent=4)
        np.save(save_dir + 'config.npy',
                convert_config_to_savable(self.config))
        np.save(save_dir + 'full_chain.npy', self.sampler.get_chain())
        np.save(save_dir + 'flat_chain.npy', self.sampler.get_chain(
            discard=int(self.nsteps * self.remove_frac), thin=self.thin,
            flat=True))
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


def load_chain(item='latest'):
    print('will be deleted soom please use load_chain_emcee!!\n')
    base = 'results/'
    save = 'test_emcee'
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
    print(f"done loading\naccess result with:\n{keys}")
    return result

def load_chain_emcee(item='latest'):
    base = 'results/'
    save = 'test_emcee'
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
