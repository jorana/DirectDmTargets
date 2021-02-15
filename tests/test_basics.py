import DirectDmTargets as dddm
from sys import platform
import tempfile
import os
import matplotlib.pyplot as plt
import time


def _is_windows():
    return 'win' in platform


def test_nested_simple_multinest():
    if _is_windows():
        return
    fit_class = dddm.NestedSamplerStatModel('Xe')
    fit_class.config['tol'] = 0.99
    fit_class.config['nlive'] = 10
    print(f"Fitting for parameters:\n{fit_class.config['fit_parameters']}")
    fit_class.run_multinest()
    fit_class.get_summary()


def test_nested_astrophysics_multinest():
    if _is_windows():
        return
    fit_unconstrained = dddm.NestedSamplerStatModel('Xe')
    fit_unconstrained.config['tol'] = 0.99
    fit_unconstrained.config['nlive'] = 10
    fit_unconstrained.set_fit_parameters(fit_unconstrained.known_parameters)
    print(
        f"Fitting for parameters:\n{fit_unconstrained.config['fit_parameters']}")
    fit_unconstrained.run_multinest()
    fit_unconstrained.get_summary()
    with tempfile.TemporaryDirectory() as tmpdirname:
        def _ret_temp(*args):
            return tmpdirname
        dddm.utils.get_result_folder = _ret_temp
        fit_unconstrained.save_results()
        save_as = fit_unconstrained.get_save_dir()
        import warnings
        warnings.warn(save_as)
        r = dddm.nested_sampling.load_multinest_samples_from_file(save_as)
        dddm.nested_sampling.multinest_corner(r)


def test_nested_simple_multinest_earth_shielding():
    if _is_windows():
        return
    else:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
    fit_class = dddm.NestedSamplerStatModel('Xe')
    fit_class.config['tol'] = 0.1
    fit_class.config['nlive'] = 3
    fit_class.config['earth_shielding'] = True
    fit_class.config['max_iter'] = 3
    print(f"Fitting for parameters:\n{fit_class.config['fit_parameters']}")
    fit_class.run_multinest()
    time.sleep(10)
    if rank == 0:
        # prevent strange race issues.
        fit_class.get_summary()


def test_nested_simple_nestle_earth_shielding():
    fit_class = dddm.NestedSamplerStatModel('Xe')
    fit_class.config['sampler'] = 'nestle'
    fit_class.config['tol'] = 0.9999
    fit_class.config['nlive'] = 3
    fit_class.config['earth_shielding'] = True
    print(f"Fitting for parameters:\n{fit_class.config['fit_parameters']}")
    fit_class.run_nestle()
    fit_class.get_summary()


def test_nested_astrophysics_nestle():
    if _is_windows():
        return
    fit_unconstrained = dddm.NestedSamplerStatModel('Xe')
    fit_unconstrained.config['sampler'] = 'nestle'
    fit_unconstrained.config['tol'] = 0.99
    fit_unconstrained.config['nlive'] = 10
    fit_unconstrained.set_fit_parameters(fit_unconstrained.known_parameters)
    print(
        f"Fitting for parameters:\n{fit_unconstrained.config['fit_parameters']}")
    fit_unconstrained.run_nestle()
    fit_unconstrained.get_summary()


def test_emcee():
    fit_class = dddm.MCMCStatModel('Xe')
    fit_class.nwalkers = 10
    fit_class.nsteps = 20

    with tempfile.TemporaryDirectory() as tmpdirname:
        fit_class.run_emcee()
        fit_class.show_corner()
        fit_class.show_walkers()
        fit_class.save_results(save_to_dir=tmpdirname)
        save_dir = fit_class.config['save_dir']
        r = dddm.emcee_applications.load_chain_emcee(
            override_load_from=save_dir)
        dddm.emcee_applications.emcee_plots(r)
        plt.clf()
        plt.close()
