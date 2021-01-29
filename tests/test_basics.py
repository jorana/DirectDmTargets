import DirectDmTargets as dddm
from sys import platform


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
    fit_class.nwalkers = 4
    fit_class.nsteps = 10
    fit_class.run_emcee()
