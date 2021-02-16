import DirectDmTargets as dddm
import tempfile
import matplotlib.pyplot as plt
from .test_multinest_shielded import _is_windows
import logging
log = logging.getLogger()


def test_nested_simple_multinest():
    if _is_windows():
        return
    fit_class = dddm.NestedSamplerStatModel('Xe')
    fit_class.config['tol'] = 0.1
    fit_class.config['nlive'] = 10
    print(f"Fitting for parameters:\n{fit_class.config['fit_parameters']}")
    fit_class.run_multinest()
    fit_class.get_summary()


def test_nested_astrophysics_multinest():
    if _is_windows():
        return
    fit_unconstrained = dddm.NestedSamplerStatModel('Xe')
    fit_unconstrained.config['tol'] = 0.1
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
        fit_unconstrained.check_did_run()
        fit_unconstrained.check_did_save()
        r = dddm.nested_sampling.load_multinest_samples_from_file(save_as)
        dddm.nested_sampling.multinest_corner(r)
        fit_unconstrained.empty_garbage()
        fit_unconstrained.show_corner()
        plt.clf()


def test_nested_astrophysics_nestle():
    fit_unconstrained = dddm.NestedSamplerStatModel('Xe')
    fit_unconstrained.config['sampler'] = 'nestle'
    fit_unconstrained.config['tol'] = 0.1
    fit_unconstrained.config['nlive'] = 10
    fit_unconstrained.config['max_iter'] = 1
    fit_unconstrained.set_fit_parameters(fit_unconstrained.known_parameters)
    print(
        f"Fitting for parameters:\n{fit_unconstrained.config['fit_parameters']}")
    fit_unconstrained.run_nestle()
    fit_unconstrained.get_summary()


def test_nestle():
    stats = dddm.NestedSamplerStatModel('Xe')
    stats.config['sampler'] = 'nestle'
    stats.config['tol'] = 0.1
    stats.config['nlive'] = 30
    # stats.config['max_iter'] = 1
    print('print info')
    stats.print_before_run()
    print('Start run')
    stats.run_nestle()
    print('Save results')
    stats.save_results()
    print('Empty garbade')
    stats.empty_garbage()
    print('Show corner')
    stats.show_corner()
    plt.close()
    plt.clf()
    print('Save & show again')
    # Deprecate this function?
    stats.get_tmp_dir()
    stats.get_save_dir()
    plt.close()
    plt.clf()
