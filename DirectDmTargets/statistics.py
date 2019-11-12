from .detector import *
from .halo import *
import numericalunits as nu


def get_priors():
    """
    :return: dictionary of priors, type and values
    """
    priors = \
        {
            'log_mass':
                {'range': [0.1, 3], 'prior_type': 'flat'},
            'log_cross_section':
                {'range': [-46, -42], 'prior_type': 'flat'},
            'density':
                {'range': [0.001, 0.9], 'prior_type': 'gauss', 'mean': 0.4,
                 'std': 0.1},
            'v_0':
                {'range': [80, 380], 'prior_type': 'gauss', 'mean': 230,
                 'std': 30},
            'v_esc':
                {'range': [379, 709], 'prior_type': 'gauss', 'mean': 544,
                 'std': 33},
            'k':
                {'range': [0.5, 3.5], 'prior_type': 'flat'}
        }
    for key in priors.keys():
        param = priors[key]
        if param['prior_type'] == 'flat':
            param['param'] = param['range']
            param['dist'] = lambda x: flat_prior_distribution(x)
        elif param['prior_type'] == 'gauss':
            param['param'] = param['mean'], param['std']
            param['dist'] = lambda x: gaus_prior_distribution(x)
    return priors


def get_prior_list():
    return ['mw', 'sigma', 'v_0', 'v_esc', 'rho_0']


def get_param_list():
    return ['log_mass', 'log_cross_section', 'v_0', 'v_esc', 'density']


class StatModel:
    def __init__(self, detector_name):
        """
        Statistical model used for Bayesian interference of detection in
        multiple experiments.
        :param detector_name: name of the detector (e.g. Xe)
        """

        assert (type(detector_name) is str and
                detector_name in detectors.keys()), "Invalid detector name"
        self.config = dict()
        self.config['detector'] = detector_name
        self.config['prior'] = get_priors()
        self.config['poisson'] = False
        self.config['v_0'] = 230
        self.config['v_esc'] = 544
        self.config['rho_0'] = 0.4
        self.config['n_energy_bins'] = 10
        print(
            f"StatModel::\tinitialized for {detector_name} detector. See "
            f"print(stat_model) for default settings")
        self.set_default()

    def __str__(self):
        return f"StatModel::for {self.config['detector']} detector. For info " \
               f"see the config file:\n{self.config}"

    def set_benchmark(self, mw=50, sigma=-45, verbose=True):
        """
        Set up the benchmark used in this statistical model. Likelihood of other
        models can be evaluated for this 'truth'

        :param mw: mass of benchmark wimp in GeV. log10(mass) will be saved to
        config
        :param sigma: cross-secontion of wimp in cm^2. log10(sigma) will be
        saved to config
        :param verbose: bool, if True add print statements
        """
        if verbose:
            print(f"StatModel::\ttaking log10 of mass of {mw}")
        self.config['mw'] = np.log10(mw)
        self.config['sigma'] = sigma
        if not ((mw == 50) and (sigma == -45)):
            print("StatModel::\tre-evaluate benchmark")
            self.eval_benchmark()

    def set_models(self, halo_model='default', spec='default'):
        """
        Update the config with the required settings
        :param halo_model: The halo model used
        :param spec: class used to generate the response of the spectrum in the
        detector
        """
        self.config[
            'halo_model'] = halo_model if halo_model != 'default' else SHM(
            v_0=self.config['v_0'] * nu.km / nu.s,
            v_esc=self.config['v_esc'] * nu.km / nu.s,
            rho_dm=self.config['rho_0'] * nu.GeV / nu.c0 ** 2 / nu.cm ** 3
        )
        self.config[
            'spectrum_class'] = spec if spec != 'default' else DetectorSpectrum
        if halo_model != 'default' or spec != 'default':
            print("StatModel::\tre-evaluate benchmark")
            self.eval_benchmark()

    def set_det_params(self):
        self.config['det_params'] = detectors[self.config['detector']]

    def set_default(self):
        self.set_benchmark(verbose=False)
        self.set_models()
        self.set_det_params()
        self.eval_benchmark()

    def check_spectrum(self):
        spectrum = self.config['spectrum_class'](
            10 ** self.config['mw'],
            10 ** self.config['sigma'],
            self.config['halo_model'],
            self.config['det_params'])
        spectrum.n_bins = self.config['n_energy_bins']
        return spectrum.get_data(poisson=self.config['poisson'])

    def eval_benchmark(self):
        # TODO make sure always evaluated before the log_probablity is evaluated
        self.benchmark_values = self.check_spectrum()['counts']

    def log_probability(self, parameter_vals, parameter_names):
        """

        :param parameter_vals: the values of the model/benchmark considered as
        the truth
        # :param parameter_values: the values of the parameters that are being
        varied
        :param parameter_names: the names of the parameter_values
        :return:
        """
        # single parameter to fit
        if type(parameter_names) == str:
            lp = self.log_prior(parameter_vals, parameter_names)

        # check the input and compute the prior
        elif len(parameter_names) > 1:
            assert len(parameter_vals) == len(
                parameter_names), f"provide enough names (" \
                                  f"{parameter_names}) for the " \
                                  f"parameters (len{len(parameter_vals)})"
            lp = np.sum([self.log_prior(*_x) for _x in
                         zip(parameter_vals, parameter_names)])
        else:
            raise TypeError(
                f"incorrect format provided. Theta should be array-like for "
                f"single value of parameter_names or Theta should be matrix-like for "
                f"array-like parameter_names. Theta, parameter_names (provided) "
                f"= {parameter_vals, parameter_names}")
        if not np.isfinite(lp):
            return -np.inf
        model = self.eval_spectrum(parameter_vals, parameter_names)

        # Compute the likelihood
        ll = log_likelihood(model, self.benchmark_values)
        if np.isnan(lp + ll):
            raise ValueError(
                f"Returned NaN from likelihood. lp = {lp}, ll = {ll}")
        return lp + ll

    def log_prior(self, value, variable_name):
        """
        Compute the prior of variable_name for a given value
        :param value: value of variable name
        :param variable_name: name of the 'value'. This name should be in the
        config of the class under the priors with a similar content as the
        priors as specified in the get_prior function.
        :return: prior of value
        """
        # For each of the priors read from the config file how the prior looks
        # like. Get the boundaries (and mean (m) and width (s) for gausian
        # distributions).
        if self.config['prior'][variable_name]['prior_type'] == 'flat':
            a, b = self.config['prior'][variable_name]['param']
            return log_flat(a, b, value)
        elif self.config['prior'][variable_name]['prior_type'] == 'gauss':
            a, b = self.config['prior'][variable_name]['range']
            m, s = self.config['prior'][variable_name]['param']
            return log_gauss(a, b, m, s, value)
        else:
            raise TypeError(
                f"unknown prior type "
                f"'{self.config['prior'][variable_name]['prior_type']}'"
                f", choose either gauss or flat")

    def eval_spectrum(self, values, parameter_names):
        """
        For given values and parameter names, return the spectrum one would have
        with these parameters. The values and parameter names should be array
        like objects of the same length. Usually, one fits either two
        ('log_mass', 'log_cross_section') or five parameters ('log_mass',
        'log_cross_section', 'v_0', 'v_esc', 'density').
        :param values: array like object of
        :param parameter_names: names of parameters
        :return: a spectrum as specified by the parameter_names
        """
        default_order = ['log_mass', 'log_cross_section', 'v_0', 'v_esc',
                         'density', 'k']
        if type(parameter_names) is str:
            raise NotImplementedError(
                f"Trying to fit a single parameter ({parameter_names}), such a "
                f"feature is not implemented.")
        if len(parameter_names) == 2:
            x0, x1 = check_shape(values)
            if (parameter_names[0] is 'log_mass'
                    and parameter_names[1] is 'log_cross_section'):
                # This is the right order
                pass
            elif (parameter_names[1] is 'log_mass'
                  and parameter_names[0] is 'log_cross_section'):
                x0, x1 = x1, x0
            else:
                raise NotImplementedError(
                    f"Trying to fit two parameters ({parameter_names}), this "
                    f"is not implemented.")
            spectrum = self.config['spectrum_class'](
                10 ** x0,
                10 ** x1,
                self.config['halo_model'], self.config['det_params'])
            spectrum.n_bins = self.config['n_energy_bins']
            return spectrum.get_data(poisson=False)
        elif len(parameter_names) == 5 or len(parameter_names) == 6:
            if not parameter_names == default_order[:len(parameter_names)]:
                raise NameError(
                    f"The parameters are not in correct order. Please insert"
                    f"{default_order[:len(parameter_names)]} rather than "
                    f"{parameter_names}.")

            checked_values = check_shape(values)
            if len(parameter_names) == 5:
                fit_shm = SHM(
                    v_0=checked_values[2] * nu.km / nu.s,
                    v_esc=checked_values[3] * nu.km / nu.s,
                    rho_dm=checked_values[4] * nu.GeV / nu.c0 ** 2 / nu.cm ** 3)
            if len(parameter_names) == 6:
                raise NotImplementedError(
                    f"Currently not yet ready to fit for {parameter_names}")

            spectrum = self.config['spectrum_class'](10 ** checked_values[0],
                                                     10 ** checked_values[1],
                                                     fit_shm,
                                                     self.config['det_params'])
            spectrum.n_bins = self.config['n_energy_bins']
            result = spectrum.get_data(poisson=False)

            if np.any(result['counts'] < 0):
                raise ValueError(
                    f"statistics.py::Finding negative rates. Presumably v_esc"
                    f" is too small ({values[3]})\nFull dump of parameters:\n"
                    f"{parameter_names} = {values}.\nIf this occurs, one or "
                    f"more priors might not be constrained correctly.")
            return result
        elif len(parameter_names) > 2 and not len(parameter_names) == 5 and \
                not len(parameter_names) == 6:
            raise NotImplementedError(
                f"Not so quickly cowboy, before you code fitting "
                f"{len(parameter_names)} parameters or more, first code it! "
                f"You are now trying to fit {parameter_names}. Make sure that "
                f"you are not using forcing a string in this part of the code)")
        else:
            raise NotImplementedError(
                f"Something strange went wrong here. Trying to fit for the"
                f"parameter_names = {parameter_names}")


def log_fact(n):
    return np.log(np.math.gamma(n + 1))


def approx_log_fact(n):
    """take the approximate logarithm of factorial n for large n

    :param n: the number n
     :return:  ln(n!)"""
    assert n >= 0, f"Only take the logarithm of n>0. (n={n})"

    # if n is small, there is no need for approximation
    if n < 10:
        # gamma equals factorial for x -> x +1 & returns results for non-int
        return log_fact(n)

    # Stirling's approx. <https://nl.wikipedia.org/wiki/Formule_van_Stirling>
    # return n * np.log(n) - n
    return (n * np.log(n) - n + 0.5 * np.log(2 * np.pi * n)
            + 1 / (12 * n) - 1 / (360 * (n ** 3)) + 1 / (1260 * (n ** 5))
            - 1 / (1680 * (n ** 7)))


def log_likelihood_function(nb, nr):
    """return the ln(likelihood) for Nb expected events and Nr observed events

    :param nb: expected events
    :param nr: observed events
    :return: ln(likelihood)
    """
    #TODO Test if this is needed
    # # No need for approximating very small values of N
    if ((nr < 5 and nb < 5) or
        (nr < 1 and nb < 10)):
         return np.log(((nr ** nb) / np.math.gamma(nb + 1)) * np.exp(-nr))
    # https://www.wolframalpha.com/input/?i=simplify+ln%28R%5Eb+%2F+b%21+exp%28-R%29%29
    return nb * np.log(nr) - approx_log_fact(nb) - nr


def log_likelihood(model, y):
    """
    :param model: pandas dataframe containing the number of counts in bin i
    :param y: the number of counts in bin i
    :return: sum of the log-likelihoods of the bins
    """

    assert len(y) == model.shape[
        0], f"Data and model should be of same dimensions (now " \
            f"{len(y), model.shape[0]})"
    assert_str = f"please insert pd.dataframe for model ({type(model)})"
    assert type(model) == pd.DataFrame, assert_str

    # TODO should start at 0 right?
    res = 0
    for i in range(len(y)):
        Nr = y[i]
        Nb = model['counts'][i]
        # https://www.wolframalpha.com/input/?i=simplify+ln%28R%5Eb+%2F+b%21+exp%28-b%29%29

        res_bin = log_likelihood_function(Nb, Nr)
        if np.isnan(res_bin):
            # TODO, return scientific notation for Nb and Nr 
            raise ValueError(
                f"Returned NaN in bin {i}. Below follows data dump.\n"
                f"i = {i}, Nb, Nr = {Nb, Nr}\n"
                f"res_bin {res_bin}\n"
                f"log(Nr) = {np.log(Nr)}, Nb! = {approx_log_fact(Nb)}\n"
                f"log_likelihood: {log_likelihood_function(Nb, Nr)}\n")
        if not np.isfinite(res_bin):
            return -np.inf
        res += res_bin
    return res


def not_nan_inf(x):
    """
    :param x: float or array
    :return: array of True and/or False indicating if x is nan/inf
    """
    if np.shape(x) == () and x is None:
        x = np.nan
    try:
        return np.isnan(x) ^ np.isinf(x)
    except TypeError:
        return np.array([not_nan_inf(xi) for xi in x])


def masking(x, mask):
    """
    :param x: float or array
    :param mask: array of True and/or False
    :return: x[mask]
    """
    assert len(x) == len(
        mask), f"match length mask {len(mask)} to length array {len(x)}"
    try:
        return x[mask]
    except TypeError:
        return np.array([x[i] for i in range(len(x)) if mask[i]])


def remove_nan(x, maskable=False):
    """
    :param x: float or array
    :param maskable: array to take into consideration when removing NaN and/or
    inf from x
    :return: x where x is well defined (not NaN or inf)
    """
    if type(maskable) is not bool:
        assert len(x) == len(maskable), (f"match length maskable "
                                         f"({len(maskable)}) to length array "
                                         f"({len(x)})")
    if type(maskable) is bool and maskable is False:
        mask = ~not_nan_inf(x)
        return masking(x, mask)
    else:
        return masking(x, ~not_nan_inf(maskable) ^ not_nan_inf(x))


def flat_prior_distribution(_range):
    return np.random.uniform(_range[0], _range[1])


def gaus_prior_distribution(_param):
    mu, sigma = _param
    return np.random.normal(mu, sigma)


def check_shape(xs):
    """
    :param xs: values
    :return: flat array of values
    """
    if not len(xs) > 0:
        raise TypeError(
            f"Provided incorrect type of {xs}. Takes either np.array or list")
    if not type(xs) == np.array:
        xs = np.array(xs)
    for i, x in enumerate(xs):
        if np.shape(x) == (1,):
            xs[i] = x[0]
    return xs


def log_flat(a, b, x):
    """
    Return a flat prior as function of x in log space
    :param a: lower bound
    :param b: upper bound
    :param x: value
    :return: 0 for x in bound, -np.inf otherwise
    """
    try:
        if a < x < b:
            return 0
        else:
            return -np.inf
    except ValueError:
        result = np.zeros(len(x))
        mask = (x > a) & (x < b)
        result[~mask] = -np.inf
        return result


def log_gauss(a, b, mu, sigma, x):
    """
    Return a gaussian prior as function of x in log space
    :param a: lower bound
    :param b: upper bound
    :param mu: mean of gauss
    :param sigma: std of gauss
    :param x: value to evaluate
    :return: log prior of x evaluated for gaussian (given by mu and sigma) if in
    between the bounds
    """
    try:
        # for single values of x
        if a < x < b:
            return -0.5 * np.sum(
                (x - mu) ** 2 / (sigma ** 2) + np.log(sigma ** 2))
        else:
            return -np.inf
    except ValueError:
        # for array like objects return as follows
        result = np.zeros(len(x))
        mask = (x > a) & (x < b)
        result[~mask] = -np.inf
        result[mask] = -0.5 * np.sum(
            (x - mu) ** 2 / (sigma ** 2) + np.log(sigma ** 2))
        return result
