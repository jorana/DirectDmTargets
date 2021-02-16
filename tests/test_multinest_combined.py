import DirectDmTargets as dddm
from .test_multinest_shielded import _is_windows
import logging

log = logging.getLogger()

def test_nested_simple_multinest_earth_shielding():
    if _is_windows():
        return
    dddm.experiment['Combined'] = {'type': 'combined'}
    stats = dddm.CombinedInference(
        ('Xe', 'Ge'),
        'Combined',
        do_init=False)
    update = {}
    update['prior'] = dddm.statistics.get_priors("Evans_2019")
    update['halo_model'] = dddm.SHM()
    update['type'] = 'SI'
    stats.config.update(update)
    stats.copy_config(list(update.keys()))
    stats.print_before_run()
    stats.config['tol'] = 0.1
    stats.config['nlive'] = 5
    print(f"Fitting for parameters:\n{stats.config['fit_parameters']}")
    stats.run_multinest()
    stats.save_results()
    stats.save_sub_configs()
