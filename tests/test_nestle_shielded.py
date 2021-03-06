import logging

import DirectDmTargets as dddm

log = logging.getLogger()


def test_nested_simple_nestle_earth_shielding():

    fit_class = dddm.NestedSamplerStatModel('Xe')
    fit_class.config['sampler'] = 'nestle'
    fit_class.config['tol'] = 0.1
    fit_class.config['nlive'] = 3
    fit_class.config['max_iter'] = 1
    fit_class.config['earth_shielding'] = True
    fit_class.config['save_intermediate'] = True
    log.info(f"Fitting for parameters:\n{fit_class.config['fit_parameters']}")
    fit_class.set_models()
    fit_class.print_before_run()
    fit_class.run_nestle()
    fit_class.get_summary()
