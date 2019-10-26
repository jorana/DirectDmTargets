import numpy as np
import wimprates as wr
from scipy.integrate import quad as scipy_int
from .halo import GenSpectrum, get_bins
from numba import jit
import numba
import pandas as pd

def det_res_Xe(E):
    return 0.6 * np.sqrt(E)


def det_res_Ar(E):
    return 0.7 * np.sqrt(E)


def det_res_Ge(E):
    return np.sqrt(0.3 ** 2 + (0.06 ** 2) * E)


benchmark = {'mw': 50, 'sigma_nucleon': 10e-45}
detectors = {
    'Xe': {'exp': 5, 'cut_eff': 0.8, 'nr_eff': 0.5, 'E_thr': 10, 'res': det_res_Xe},
    'Ar': {'exp': 3, 'cut_eff': 0.8, 'nr_eff': 0.9, 'E_thr': 10, 'res': det_res_Ar},
    'Ge': {'exp': 10, 'cut_eff': 0.8, 'nr_eff': 0.8, 'E_thr': 30, 'res': det_res_Ge}}

for name in detectors.keys():
    detectors[name]['exp_eff'] = (detectors[name]['exp'] *
                                  detectors[name]['cut_eff'] *
                                  detectors[name]['nr_eff'])
    print(f"calculating effective efficancy for {name} detector done")


# def dr_eff(E, sigma_res, dr_de=wr.rate_wimp_std,
#            kwargs={'mw': 50, 'sigma_nucleon': 1e-45}):
#     """Smears the WIMP spectrum_simple with the detector resolution.
#
#     :param E: energy (in GeV)
#     :param sigma_res: function of E for calculating the detector resolution
#     :param dr_de: function of the WIMP rate (unsmeared)
#     :param kwargs: kwargs for ''dr_de''
#     by default this is set to a 50 GeV wimp with coressection 10e-45
#
#     returns: WIMP rate smeared by detector resolution
#     """
#
#     assert type(E) != np.ndarray, "Only single valued energies are allowed"
#
#     def det_smear(E, Eprime):
#         return (np.exp(-(E - Eprime) ** 2 / (2 * sigma_res(Eprime) ** 2))
#                 / (np.sqrt(2 * np.pi) * sigma_res(Eprime)))
#
#     f = lambda Eprime: dr_de(Eprime, **kwargs) * det_smear(E, Eprime)
#
#     # TODO: is this sufficient?
#     E_max = 10 * E
#     result, _ = scipy_int(f, 0, E_max)
#     return result
#
#
# # TODO I think that this function is bad (takes forever)
# def N_r(E1, E2, e_eff, smearing=True, kwargs={'mw': 50, 'sigma_nucleon': 1e-45}):
#     """Returns integral of given energy bin
#     """
#     if smearing:
#         f = lambda E: dr_eff(E, det_res_Xe)
#         res, _ = scipy_int(f, E1, E2)
#     #         assert res/_ > 1000, "integrateion errors too big"
#
#     else:
#         f = lambda E: wr.rate_wimp_std(E, **kwargs)
#         res, _ = scipy_int(f, E1, E2)
#     #         assert res/_ > 1000, "integrateion errors too big"
#     return e_eff * res
#
# # def N_r(E1, E2, e_eff, kwargs = {'mw':50, 'sigma_nucleon':1e-45}):
# #     f = lambda E: wr.rate_wimp_std(E, **kwargs)
# #     res, _ = scipy_int(f, E1, E2)
# #     assert res/_ > 1000, "integrateion errors too big"
# #     return e_eff * res
# # @jit
# # def _n_integrate(rate, eval_res, bin_width):
# #         """
# #
# #         :param rate: np.array of rates computed at the center of energybins
# #         :param eval_res: the evaluated detector resolution at the energy where the rate is
# #         :param bin_width: either array or scalar of the binwidth(s) of the energy bins wherein the
# #         rate is calculated
# #         :return: spectrum smeared with the energy resolution of the detector
# #         """
# #         if not ((type(bin_width) == np.ndarray and len(bin_width) == len(rate))
# #                 or np.isscalar(bin_width)):
# #             raise TypeError(f"bin_width is not  of correct type, should be either scalar like or"
# #                             f"of the same length as the rate. bin_width = {bin_width}")
# #         if not len(eval_res) == len(rate):
# #             raise TypeError(f"bin_width (len = {len(eval_res)}) and rate (len = {len(rate)}) are"
# #                             f"of unequal length.")
# #         if not type(rate) == type(eval_res) == np.ndarray:
# #             raise TypeError(f"both the rate and the evaluated detector resolution should be "
# #                             f"np.arrays. Instead got {type(rate)} and {type(eval_res)} "
# #                             f"respectively.")
# #
# #         return np.sum(rate * eval_res * bin_width)

def _smear_signal(rate, energy, func):
    result = np.zeros(len(rate))
    bin_width = np.mean(np.diff(energy))
    sigma = func(energy)
    for i, E in enumerate(energy):
        result[i] = np.sum(bin_width * rate * (1 / (np.sqrt(2 * np.pi) * sigma)) * \
                           np.exp(-((E - energy)**2 / (2 * sigma**2))))
    return result

# @jit
@numba.njit
def smear_signal(rate, energy, sigma, bin_width):
    result = []
    for i in range(len(energy)):
        res = 0
        for j in range(len(rate)):
            res = res +\
                        (bin_width * rate[j] *
                          (1 / (np.sqrt(2 * np.pi) * sigma[j])) *
                           np.exp(
                               -((energy[i] - energy[j])**2 / (2 * sigma[j] **2))
                           )
                          )
        result.append(res)
    return result

class DetectorSpectrum(GenSpectrum):
    # GenSpectrum generates a number of bins (default 10), however, since an numerical integration
    # is performed in compute_detected_spectrum, this number is multiplied here.

    def __init__(self, *args):
        GenSpectrum.__init__(self, *args)
        self.rebin_factor = 10

    def __str__(self):
        return (f"DetectorSpectrum class inherited from GenSpectrum.\nSmears spectrum with detector"
               f"resolution and implements the energy threshold for the detector")

    @staticmethod
    def n_integrate(rate, eval_res, bin_width):
        """

        :param rate: np.array of rates computed at the center of energybins
        :param eval_res: the evaluated detector resolution at the energy where the rate is
        :param bin_width: either array or scalar of the binwidth(s) of the energy bins wherein the
        rate is calculated
        :return: spectrum smeared with the energy resolution of the detector
        """
        if not ((type(bin_width) == np.ndarray and len(bin_width) == len(rate))
                or np.isscalar(bin_width)):
            raise TypeError(f"bin_width is not  of correct type, should be either scalar like or"
                            f"of the same length as the rate. bin_width = {bin_width}")
        if not len(eval_res) == len(rate):
            raise TypeError(f"bin_width (len = {len(eval_res)}) and rate (len = {len(rate)}) are"
                            f"of unequal length.")
        if not type(rate) == type(eval_res) == np.ndarray:
            raise TypeError(f"both the rate and the evaluated detector resolution should be "
                            f"np.arrays. Instead got {type(rate)} and {type(eval_res)} "
                            f"respectively.")

        return np.sum(rate * eval_res * bin_width)

    def _chuck_integration(self, rates, energies, bins):
        res = np.zeros(len(bins))
        for i, bin in enumerate(bins):
            mask = (energies > bin[0]) & (energies < bin[1])
            # print(mask, energies, rates, energies[mask])
            bin_width = np.average(np.diff(energies[mask]))
            eval_res = self.detector['res']
            res[i] = self.n_integrate(rates[mask], eval_res(energies[mask]), bin_width)
        return res

    def chuck_integration(self, rates, energies, bins):
        res = np.zeros(len(bins))
        for i, bin in enumerate(bins):
            mask = (energies > bin[0]) & (energies < bin[1])
            bin_width = np.average(np.diff(energies[mask]))
            #TODO ugly
            res[i] = self.n_integrate(rates[mask], np.ones(len(rates[mask])), bin_width)
        return res

    def above_threshold(self, rates, energies):
        # TODO something smarter than just a hard cutof?
        rates[energies < self.detector['E_thr']] = 0
        return rates

    def compute_detected_spectrum(self):
        # assert self.n_bins / self.n_bins_result > 2, "binning to course for numerical integration"
        self.n_bins_result = self.n_bins
        self.n_bins = self.n_bins * self.rebin_factor
        rates = self.spectrum_simple([self.mw, self.sigma_nucleon])
        energies = self.get_bin_centers()
        rates = self.above_threshold(rates, energies)
        result_bins = get_bins(self.E_min, self.E_max, self.n_bins_result)
        # events = self.chuck_integration(rates, energies, result_bins) * self.detector['exp_eff']
        # print(energies)
        # events = smear_signal(rates, energies, self.detector['res'])

        sigma = self.detector['res'](energies)
        bin_width = np.mean(np.diff(energies))
        events = np.array(smear_signal(rates, energies, sigma, bin_width))
        events = self.chuck_integration(events, energies, result_bins)
        return events * self.detector['exp_eff']

    def get_events(self):
        """
        :return: Events (binned)
        """
        return self.compute_detected_spectrum()


    def get_poisson_events(self):
        """
        :return: events with poisson noise
        """
        return np.random.poisson(self.get_events()).astype(np.float)

    def get_data(self, poisson=True):
        """

        :param poisson: type bool, add poisson True or False
        :return: pd.DataFrame containing events binned in energy
        """
        result = pd.DataFrame()
        if poisson:
            result['counts'] = self.get_poisson_events()
        else:
            result['counts'] = self.get_events()
        bins = get_bins(self.E_min, self.E_max, self.n_bins_result)
        result['bin_centers'] = np.mean(bins, axis=1)
        result['bin_left'] = bins[:, 0]
        result['bin_right'] = bins[:, 1]
        return result
