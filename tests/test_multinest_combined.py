import DirectDmTargets as dddm
from .test_multinest_shielded import _is_windows


def test_nested_simple_multinest_earth_shielding():
    if _is_windows():
        return
    dddm.experiment['test'] = {'type': 'combined'}
    stats = dddm.CombinedInference(
        ('Xe', "Ge"),
        'test',
        do_init=False)
    stats.copy_config(stats.config.keys())
    stats.config['tol'] = 0.1
    stats.config['nlive'] = 5
    print(f"Fitting for parameters:\n{stats.config['fit_parameters']}")
    stats.run_multinest()
    stats.save_results()
    stats.save_sub_configs()
