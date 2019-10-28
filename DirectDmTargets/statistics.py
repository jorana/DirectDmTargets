# import emcee
# import scipy
from .detector import *
from .halo import *
import numericalunits as nu


# TODO
# use_SHM = SHM()


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
            param['dist'] = lambda x: flat_prior(x)
        elif param['prior_type'] == 'gauss':
            param['param'] = param['mean'], param['std']
            param['dist'] = lambda x: gaus_prior(x)
    return priors


class StatModel:
    def __init__(self, detector_name):
        self.config = dict()
        self.config['detector'] = detector_name
        self.config['prior'] = get_priors()
        self.config['poisson'] = False
        self.config['v_0'] = 220
        self.config['v_esc'] = 544
        self.config['rho_0'] = 0.3
        print(
            f"stat_model::initialized for {detector_name} detector. See "
            f"print(stat_model) for default settings")
        self.set_default()

    def __str__(self):
        return f"stat_model::for {self.config['detector']} detector. For info" \
               f" see the config file:\n{self.config}"

    def set_benchmark(self, mw=50, sigma=1e-45):
        self.config['mw'] = mw
        self.config['sigma'] = sigma
        if not ((mw == 50) and (sigma == 1e-45)):
            print("re-evaluate benchmark")
            self.eval_benchmark()
            # print(f"setting the benchmark for for Mw ({mw}) and cross-section
            # ({sigma}) to default")

    def set_models(self, halo_model='default', spec='default'):
        self.config[
            'halo_model'] = halo_model if halo_model != 'default' else SHM(
            v_0=self.config['v_0'] * nu.km / nu.s,
            v_esc=self.config['v_esc'] * nu.km / nu.s,
            rho_dm=self.config['rho_0'] * nu.GeV / nu.c0 ** 2 / nu.cm ** 3
        )
        self.config[
            'spectrum_class'] = spec if spec != 'default' else DetectorSpectrum
        if halo_model != 'default' or spec != 'default':
            print("re-evaluate benchmark")
            self.eval_benchmark()

    def set_det_params(self):
        self.config['det_params'] = detectors[self.config['detector']]

    def set_default(self):
        self.set_benchmark()
        self.set_models()
        self.set_det_params()
        self.eval_benchmark()

    def check_spectrum(self):
        spectrum = self.config['spectrum_class'](
            self.config['mw'],
            self.config['sigma'],
            self.config['halo_model'],
            self.config['det_params'])
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
                f"single value of x_names or Theta should be matrix-like for "
                f"array-like x_names. Theta, x_names (provided) "
                f"= {parameter_vals, parameter_names}")
        if not np.isfinite(lp):
            return -np.inf
        model = self.eval_spectrum(parameter_vals, parameter_names)

        ll = log_likelihood(model, self.benchmark_values)
        if np.isnan(lp + ll):
            raise ValueError(
                f"Returned NaN from likelihood. lp = {lp}, ll = {ll}")
        return lp + ll



    def log_prior(self, x, x_name):
        if self.config['prior'][x_name]['prior_type'] == 'flat':
            a, b = self.config['prior'][x_name]['param']
            if 'log' in x_name:
                x = np.log10(x)
            return log_flat(a, b, x)
        elif self.config['prior'][x_name]['prior_type'] == 'gauss':
            a, b = self.config['prior'][x_name]['range']
            m, s = self.config['prior'][x_name]['param']
            # if x < 0:
            #     print(
            #         f"finding a negative value for {x_name}, returning -np.inf")
            #     return -np.inf
            return log_gauss(a, b, m, s, x)
        else:
            raise TypeError(
                f"unknown prior type '"
                f"{self.config['prior'][x_name]['prior_type']}', choose either "
                f"gauss or flat")

    def eval_spectrum(self, values, x_names):
        default_order = ['log_mass', 'log_cross_section', 'v_0', 'v_esc',
                         'density', 'k']
        if len(x_names) == 2:
            x0, x1 = check_shape(values)
            if x_names[0] == 'log_mass' and x_names[1] == 'log_cross_section':
                pass
            elif x_names[1] == 'log_mass' and x_names[0] == 'log_cross_section':
                x0, x1 = x1, x0
            spectrum = self.config['spectrum_class'](
                x0, x1, self.config['halo_model'], self.config['det_params'])
            return spectrum.get_data(poisson=False)
        elif len(x_names) == 5 or len(x_names) == 6:
            if not x_names == default_order[:len(x_names)]:
                raise NameError(
                    f"The parameters are not in correct order. Please insert"
                    f"{default_order[:len(x_names)]} rather than {x_names}.")
            xs = check_shape(values)
            if len(x_names) == 5:
                fit_shm = SHM(v_0=xs[2] * nu.km / nu.s,
                              v_esc=xs[3] * nu.km / nu.s,
                              rho_dm=xs[4] * nu.GeV / nu.c0 ** 2 / nu.cm ** 3)
            if len(x_names) == 6:
                raise NotImplementedError(
                    f"Currently not yet ready to fit for {x_names[5]}")

            spectrum = self.config['spectrum_class'](xs[0],
                                                     xs[1],
                                                     fit_shm,
                                                     self.config['det_params'])
            result = spectrum.get_data(poisson=False)

            # TODO this is not the correct way, why does wimprates produce negative rates?
            mask = (result['counts'] < 0)
            if np.any(mask):
                print(f"Serious error, finding negative rates. Presumably v_esc"
                      f" is too small ({values[3]})")
                result['counts'][mask] = 0
            return result

        elif len(x_names) > 2 and not len(x_names) == 5 and not len(
                x_names) == 6:
            raise NotImplementedError(
                f"Not so quick cow-boy, before you code fitting {len(x_names)} "
                f"parameters or more, first code it! You are now trying to fit "
                f"{x_names}. Make sure that you are not using forcing a string "
                f"in this part of the code)")
        else:
            raise NotImplementedError(
                f"Oops this is not somewhere you want to be, "
                f"x_names = {x_names}")


def approx_log_fact(n):
    """take the approximate logarithm of factorial n for large n

    :param n: the number n
     :return:  ln(n!)"""
    assert n >= 0, f"Only take the logarithm of n>0. (n={n})"

    # if n is small, there is no need for approximation
    if n < 10:
        # gamma equals factorial for x -> x +1 & returns results for non-int
        return np.log(np.math.gamma(n + 1))

    # Stirling's approx. <https://nl.wikipedia.org/wiki/Formule_van_Stirling>
    return n * np.log(n) - n


# @numba.autojit
def log_likelihood_function(nb, nr):
    """return the ln(likelihood) for Nb expected events and Nr observed events

    :param nb: expected events
    :param nr: observed events
    :return: ln(likelihood)
    """
    # No need for approximating very small values of N
    if nr < 5 and nb < 5:
        return np.log(((nr ** nb) / np.math.gamma(nb + 1)) * np.exp(-nr))
    # https://www.wolframalpha.com/input/?i=simplify+ln%28R%5Eb+%2F+b%21+exp%28-R%29%29
    return nb * np.log(nr) - approx_log_fact(nb) - nr


# @numba.autojit
# def log_likelihood_numba(nr, nb):
#     res = 1
#     error = False
#     for i in range(len(nr)):
#         # TODO round nb_i to int is okay?
#         nr_i = nr[i]
#         nb_i = nb[i]
#
#         # https://www.wolframalpha.com/input/?i=simplify+ln%28R%5Eb+%2F+b%21+exp%28-b%29%29
#         res_bin = log_likelihood_function(nb_i, nr_i)
#         if np.isnan(res_bin):
#             error = ("Returned NaN in bin " + str(i) )# + "Below follows data dump.\n i = " + str(i) +
#                      #"nb_i, nr_i = %.1f %.1f"%(nb_i, nr_i))
#         if not np.isfinite(res_bin):
#             return -np.inf
#         res += res_bin
#     return res, error


def log_likelihood(model, y):
    """
    :param model: pandas dataframe containing the number of counts in bin i
    :param y: the number of counts in bin i
    :return: product of the likelihoods of the bins
    """

    assert len(y) == model.shape[
        0], f"Data and model should be of same dimensions (now " \
            f"{len(y), model.shape[0]})"
    assert_str = f"please insert pd.dataframe for model ({type(model)})"
    assert type(model) == pd.DataFrame, assert_str
    # TODO Also add the assertion error for x and y

    ym = model['counts']
    # res, err = log_likelihood_numba(y, ym)
    # if err:
    #     raise ValueError(err)
    # else:
    #     return res
    res = 1
    for i in range(len(y)):

        Nr = y[i]
        Nb = model['counts'][i]
        # TODO round Nb to int is okay?
        ## https://www.wolframalpha.com/input/?i=simplify+ln%28R%5Eb+%2F+b%21+exp%28-b%29%29

        res_bin = log_likelihood_function(Nb, Nr)
        if np.isnan(res_bin):
            raise ValueError(
                f"Returned NaN in bin {i}. Below follows data dump.\n"
                f"i = {i}, Nb, Nr = {Nb, Nr}\n"
                f"res_bin {res_bin}\n"
                f"log(Nr) = {np.log((Nr))}, Nb! = {approx_log_fact(Nb)}\n"
                f"log_likelihood: {log_likelihood_function(Nb, Nr)}\n"
            )
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


def flat_prior(_range):
    return np.random.uniform(_range[0], _range[1])


def gaus_prior(_param):
    mu, sigma = _param
    return np.random.normal(mu, sigma)


def check_shape(xs):
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
    try:
        if a < x < b:
            return -0.5 * np.sum(
                (x - mu) ** 2 / (sigma ** 2) + np.log(sigma ** 2))
        else:
            return -np.inf
    except ValueError:
        result = np.zeros(len(x))
        mask = (x > a) & (x < b)
        result[~mask] = -np.inf
        result[mask] = -0.5 * np.sum(
            (x - mu) ** 2 / (sigma ** 2) + np.log(sigma ** 2))
        return result
