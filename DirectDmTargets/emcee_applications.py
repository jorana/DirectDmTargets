import emcee
from multiprocessing import Pool
from .statistics import *
from .detector import *
from .halo import *

def truth():
    use_SHM = SHM()
    xe_events = DetectorSpectrum(50, 1e-45, use_SHM, detectors['Xe'])
    xe_data = xe_events.get_data(poisson = False)
    return xe_data

def fit_emcee(log_probability = log_probability_detector, nwalkers, nsteps, pos, args, pool):
    with Pool() as pool:
        nwalkers = 500
        step = 200
        pos = np.hstack(
            [50 + 3 * 10 * np.random.rand(nwalkers, 1),
             1e-45 + 1e-45 * np.random.rand(nwalkers, 1),
             230 + 3 * 30 * np.random.rand(nwalkers, 1),
             544 + 3 * 33 * np.random.rand(nwalkers, 1),
             0.4 + 3 * 0.1 * np.random.rand(nwalkers, 1)
             ])
        nwalkers, ndim = pos.shape

        sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                        dddm., args=(xe_data['bin_centers'],
                                                                             xe_data['counts'],
                                                                             ['log_mass',
                                                                              'log_cross_section',
                                                                              'v_0',
                                                                              'v_esc',
                                                                              'density'])
                                        , pool=pool
                                        )
        sampler.run_mcmc(np.abs(pos), step, progress=True);
