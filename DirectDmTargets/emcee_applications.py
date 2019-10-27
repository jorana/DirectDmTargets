import emcee
from multiprocessing import Pool
from .statistics import *
from .detector import *
from .halo import *
import emcee
import multiprocessing
import matplotlib.pyplot as plt
import corner


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
            [[self.config['prior'][param]['dist'](
                self.config['prior'][param]['param']
                ) for i in range(self.nwalkers)]
                for param in self.fit_parameters]
            ])
        for i, p in enumerate(self.fit_parameters):
            if 'log' in p:
                pos[i] = 10**pos[i]
            # #TODO workaround
            # if 'cross' in p:
            #     pos[i] = 1e-45 + 1e-45 * np.random.rand(self.nwalkers)
            # if 'mass' in p:
            #     pos[i] = 50 + 50 * np.random.rand(self.nwalkers)
        self.pos = pos.T

    def set_pos(self, use_pos=None):
        self.log['pos'] = True
        if use_pos is not None:
            print("using specified start position")
            self.pos = use_pos
            return
        nparameters = len(self.fit_parameters)
        keys = ['mw', 'sigma', 'v_0', 'v_esc', 'rho_0'][:nparameters]
        vals = [self.config.get(key) for key in keys]
        pos = np.hstack([
            val + 0.1 * val * np.random.randn(self.nwalkers, 1)
            for val in vals
            ])
        self.pos = np.abs(pos)

    def set_sampler(self):
        ndim = len(self.fit_parameters)
        self.sampler = emcee.EnsembleSampler(self.nwalkers, ndim,
                              self.log_probability,
                              args=([self.fit_parameters]),
                              threads=multiprocessing.cpu_count())
        self.log['sampler'] = True

    def run_emcee(self):
        if not self.log['sampler']:
            self.set_sampler()
        if not self.log['pos']:
            self.set_pos()
        try:
            self.sampler.run_mcmc(self.pos, self.nsteps, progress=True)
        except ValueError as e:
            print(f"MCMC did not finish due to a ValueError. Was running with\n"
                  f"pos={self.pos.shape} nsteps = {self.nsteps}, walkers = "
                  f"{self.nwalkers}, ndim = "
                  f"{len(self.fit_parameters)} for fit parameters "
                  f"{self.fit_parameters}")
            raise e
        self.log['did_run'] = True

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
        # plt.show()

    def show_corner(self):
        if not self.log['did_run']:
            self.run_emcee()
        remove_frac = 0.2
        print(f"Removing a fraction of {remove_frac} of the samples")
        flat_samples = self.sampler.get_chain(
            discard=int(self.nsteps * remove_frac), thin=15, flat=True)
        print(flat_samples.shape)
        #TODO
        # truths = [50, 1e-45, 230, 544, 0.3]
        truths = [self.config['mw'],
                  self.config['sigma'],
                  self.config['v_0'],
                  self.config['v_esc'],
                  self.config['rho_0']
                  ]

        fig = corner.corner(flat_samples, labels=self.fit_parameters,
                            truths=truths[:len(self.fit_parameters)])
        # plt.show()