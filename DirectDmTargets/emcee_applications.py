import emcee
from multiprocessing import Pool
from .statistics import *
from .detector import *
from .halo import *
import emcee
import multiprocessing
import matplotlib.pyplot as plt
import corner
# def truth():
#     use_SHM = SHM()
#     xe_events = DetectorSpectrum(50, 1e-45, use_SHM, detectors['Xe'])
#     xe_data = xe_events.get_data(poisson = False)
#     return xe_data
# nwalkers = 500
# def fit_emcee(log_probability = log_probability_detector,
#               nwalkers = 500,
#               nsteps = 200,
#               pos = np.hstack(
#             [50 + 3 * 10 * np.random.rand(nwalkers, 1),
#              1e-45 + 1e-45 * np.random.rand(nwalkers, 1),
#              230 + 3 * 30 * np.random.rand(nwalkers, 1),
#              544 + 3 * 33 * np.random.rand(nwalkers, 1),
#              0.4 + 3 * 0.1 * np.random.rand(nwalkers, 1)
#              ]),
#               args = args=(xe_data['bin_centers'],
#                                                                              xe_data['counts'],
#                                                                              ['log_mass',
#                                                                               'log_cross_section',
#                                                                               'v_0',
#                                                                               'v_esc',
#                                                                               'density']),
#               pool = False):
#     with Pool() as pool:
#         nwalkers, ndim = pos.shape
#
#         sampler = emcee.EnsembleSampler(nwalkers, ndim,
#                                         log_probability, )
#                                         , pool=pool
#                                         )
#         sampler.run_mcmc(np.abs(pos), step, progress=True);


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
        self.sampler = False
        self.log = {'sampler': False, 'did_run': False}

    def set_fit_parameters(self, params):
        if not type(params) == list:
            raise TypeError("Set the parameter names in a list of strings")
        for param in params:
            if param not in self.known_parameters:
                raise NotImplementedError(f"{param} does not match any of the known parameters "
                                          f"try any of {self.known_parameters}")
        if not params == self.known_parameters[:len(params)]:
            raise NameError(f"The parameters are not input in the correct order. Please insert"
                            f"{self.known_parameters[:len(params)]} rather than {params}.")
        self.fit_parameters = params

    def set_pos(self):
        pos = np.hstack([
            [[self.config['prior'][param]['dist'](
                self.config['prior'][param]['param']
            )
             for i in range(self.nwalkers)]
            for param in self.fit_parameters]
            ])
        for i, p in enumerate(self.fit_parameters):
            if 'log' in p:
                pos[i] = 10**pos[i]
            #TODO workaround
            if 'cross' in p:
                pos[i] = 1e-45 + 1e-45 * np.random.rand(self.nwalkers)
            if 'mass' in p:
                pos[i] = 50 + 50 * np.random.rand(self.nwalkers)
        return pos.T

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
        pos = self.set_pos()
        try:
            self.sampler.run_mcmc(pos, self.nsteps, progress=True)
        except ValueError as e:
            print(f"MCMC did not finish due to a ValueError. Was running with\npos={pos.shape} "
                  f"nsteps = {self.nsteps}, walkers = {self.nwalkers}, ndim = "
                  f"{len(self.fit_parameters)} for fitparamters {self.fit_parameters}")
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
                  230,
                  544,
                  0.3
                  ]

        fig = corner.corner(flat_samples, labels=self.fit_parameters,
                            truths=truths[:len(self.fit_parameters)]);
        # plt.show()