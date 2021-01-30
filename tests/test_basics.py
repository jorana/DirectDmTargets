import DirectDmTargets as dddm
from sys import platform
import tempfile
import os


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
        r = dddm.nested_sampling.load_multinest_samples_from_file(save_as)
        dddm.nested_sampling.multinest_corner(r)


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
    if _is_windows():
        return
    fit_class = dddm.MCMCStatModel('Xe')
    fit_class.nwalkers = 10
    fit_class.nsteps = 20

    with tempfile.TemporaryDirectory() as tmpdirname:
        fit_class.run_emcee()
        fit_class.show_corner()
        fit_class.show_walkers()
        # print(tmpdirname)
        # fit_class.save_results()
        # r = dddm.emcee_applications.load_chain_emcee(base=tmpdirname, load_from='.')
        # dddm.emcee_applications.emcee_plots(r)
