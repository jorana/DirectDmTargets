"""Introduce detector effects into the expected detection spectrum"""

import numba
import numpy as np
import pandas as pd

from .halo import GenSpectrum, get_bins


def det_res_Xe(E):
    """
    :param E: recoil energy (in keV)
    :return: detector resolution for Xe detector
    """
    return 0.6 * np.sqrt(E)


def det_res_Ar(E):
    """
    :param E: recoil energy (in keV)
    :return: detector resolution for Ar detector
    """
    return 0.7 * np.sqrt(E)


def det_res_Ge(E):
    """
    :param E: recoil energy (in keV)
    :return: detector resolution for Ge detector
    """
    return np.sqrt(0.3 ** 2 + (0.06 ** 2) * E)


def det_res_CDMS(E):
    # https: // arxiv.org / pdf / 1808.09098.pdf
    """
        :param E: recoil energy (in keV)
        :return: detector resolution for Ge detector
        """
    sigma_e = 10. / 1e3  # eV to keV
    param_a = 5.0e-3  # dimensionless
    param_b = 0.85 / 1e3  # eV to keV
    # TODO does this give the right dimensions?
    return np.sqrt(sigma_e ** 2 +
                   param_b * E +
                   (param_a * E) ** 2)


def det_res_XENON1T(E):
    # TODO
    return det_res_Xe(E)


def det_res_DarkSide(E):
    # TODO
    return det_res_Ar(E)


def migdal_background_XENON1T(e_min, e_max, nbins):
    '''
     :param nbins: number of bins
    :return: detector resolution for Ge detector

    '''
    # TODO really ugly
    # background in XENON1T
    # p. 140 https://pure.uva.nl/ws/files/31193425/Thesis.pdf
    bg_rate = 1.90 * 1.0e-4  # kg day / keV
    conv_units = 1.0e-3 * (1. / 365.25)  # Tonne year
    # Assume flat background over entire energy range
    # True to first order below 200 keV

    return np.full(nbins, bg_rate * conv_units)



@numba.jit(nopython=True)
def migdal_background_CDMS(e_min, e_max, nbins):
    '''
     :param E: recoil energy (in keV)
    :return: detector resolution for Ge detector

    '''
    bins = np.linspace(e_min, e_max, nbins)
    # res = [CDMS_background_functions(bin) for bin in bins]
    res = []
    conv_units = 1.0e-3 * (1 / 365.25)
    for i in range(nbins):
        res.append(
            CDMS_background_functions(bins[i]) * conv_units)


    return np.array(res)


@numba.jit(nopython=True)
def CDMS_background_functions(E):
    # background in XENON1T
    # p. 140 https://pure.uva.nl/ws/files/31193425/Thesis.pdf
    if E < 3:  # keV
        return 0.9  # kg day / keV
    elif E < 5:
        return 0.1
    elif E < 8:
        return 0.01
    else:
        return 0.01


def migdal_background_Darkside(e_min, e_max, nbins):
    # TODO this is crude
    # background in XENON1T
    # p. 140 https://pure.uva.nl/ws/files/31193425/Thesis.pdf
    return 1e4 * migdal_background_XENON1T(e_min, e_max, nbins)


# TODO
# def threshold_function_Xe(E)
#     """
#     :param E: recoil energy (in keV)
#     :return: detector resolution for Ge detector
#     """
#     return np.sqrt(0.3 ** 2 + (0.06 ** 2) * E)

# Set the default benchmark for a 50 GeV WIMP with a cross-section of 1e-45 cm^2
benchmark = {'mw': 50., 'sigma_nucleon': 1e-45}

# Set up a dictionary of the different detectors
# Each experiment below lists:
# Name :{Interaction type (type0, exposure [ton x yr] (exp.), cut efficiency (cut_eff),
# nuclear recoil acceptance (nr_eff), energy threshold [keV] (E_thr), resolution function (res)
experiment = {
    'Xe': {'material': 'Xe', 'type': 'SI', 'exp': 5., 'cut_eff': 0.8, 'nr_eff': 0.5, 'E_thr': 10.,
           'res': det_res_Xe},
    'Ge': {'material': 'Ge', 'type': 'SI', 'exp': 3., 'cut_eff': 0.8, 'nr_eff': 0.9, 'E_thr': 10.,
           'res': det_res_Ge},
    'Ar': {'material': 'Ar', 'type': 'SI', 'exp': 10., 'cut_eff': 0.8, 'nr_eff': 0.8, 'E_thr': 30.,
           'res': det_res_Ar},
    'Xe_migd': {
        'material': 'Xe',
        'type': 'migdal',
        'exp': 5 * 5,  # aim for 2025 (5 yr * 5 ton)
        'cut_eff': 0.8,
        'nr_eff': 0.5,
        'E_thr': 1.4,  # https://arxiv.org/abs/1907.12771
        'location': "XENON",
        'res': det_res_XENON1T,
        'bg_func': migdal_background_XENON1T},
    'Ge_migd': {
        'material': 'Ge',
        'type': 'migdal',
        # TODO 100 kg yr (for 56 + 44 Ge in Table I.)
        'exp': 100 * 1.e-3,
        # https://www.slac.stanford.edu/exp/cdms/ScienceResults/Publications/PhysRevD.95.082002.pdf
        'cut_eff': 0.8,
        'nr_eff': 0.9,
        'E_thr': 70. / 1e3,  # Assume similar to CDMSlite https://arxiv.org/pdf/1808.09098.pdf
        "location":"SUF",
        'res': det_res_CDMS,
        'bg_func': migdal_background_CDMS},
    'Ar_migd': {
        'material': 'Ar',
        'type': 'migdal',
        'exp': 10. * 5,
        'cut_eff': 0.8,
        'nr_eff': 0.8,
        # TODO otherwise no results at all, but they really don't have a 5 keV threshold
        'E_thr': 3.,
        "location" : "XENON",
        'res': det_res_DarkSide,
        'bg_func': migdal_background_Darkside}}

# And calculate the effective exposure for each
for name in experiment.keys():
    experiment[name]['exp_eff'] = (experiment[name]['exp'] *
                                   experiment[name]['cut_eff'] *
                                   experiment[name]['nr_eff'])
    experiment[name]['name'] = name
    print(f"calculating effective efficiency for {name} detector done")


@numba.njit
def smear_signal(rate, energy, sigma, bin_width):
    """

    :param rate: counts/bin
    :param energy: energy bin_center
    :param sigma: energy resolution
    :param bin_width: should be scalar of the bin width
    :return: the rate smeared with the specified energy resolution at given
    energy

    This function takes a binned DM-spectrum and takes into account the energy
    resolution of the detector. The rate, energy and resolution should be arrays
    of equal length. The the bin_width
    """
    result = []
    for i in range(len(energy)):
        res = 0
        for j in range(len(rate)):
            res = res + (
                    bin_width *
                    rate[j] *
                    (1 / (np.sqrt(2 * np.pi) * sigma[j])) *
                    np.exp(
                        -((energy[i] - energy[j]) ** 2 / (2 * sigma[j] ** 2)))
            )
        result.append(res)
    return result


class DetectorSpectrum(GenSpectrum):
    def __init__(self, *args):
        GenSpectrum.__init__(self, *args)
        # GenSpectrum generates a number of bins (default 10), however, since a
        # numerical integration is performed in compute_detected_spectrum, this
        # number is multiplied here.
        self.rebin_factor = 10
        self.add_background = self.experiment['type'] == 'migdal'

    def __str__(self):
        return (f"DetectorSpectrum class inherited from GenSpectrum.\nSmears "
                f"spectrum with detector resolution and implements the energy "
                f"threshold for the detector")

    @staticmethod
    def chuck_integration(rates, energies, bins):
        """
        :param rates: counts/bin
        :param energies: energy bin_center
        :param bins: two-dimensional array of the bin-boundaries wherein the
        energies should be integrated
        :return: the re-binned number of counts/bin specified by the two-
        dimensional array bins
        """
        res = np.zeros(len(bins))
        for i, bin_i in enumerate(bins):
            # bin_i should be right bin and left bin
            mask = (energies > bin_i[0]) & (energies < bin_i[1])
            bin_width = np.average(np.diff(energies[mask]))
            res[i] = np.sum(rates[mask] * bin_width)
        return res

    def above_threshold(self, rates, energies):
        # TODO something smarter than just a hard cutoff?
        rates[energies < self.experiment['E_thr']] = 0
        return rates

    def compute_detected_spectrum(self):
        """

        :return: spectrum taking into account the detector properties
        """
        # The numerical integration requires finer binning, therefore compute a
        # spectrum at finer binning than the number of bins the result should be
        # in.
        self.n_bins_result = self.n_bins
        self.n_bins = self.n_bins * self.rebin_factor
        rates = self.spectrum_simple([self.mw, self.sigma_nucleon])
        if self.add_background:
            rates += self.experiment['bg_func'](self.E_min, self.E_max, self.n_bins)
        energies = self.get_bin_centers()
        rates = self.above_threshold(rates, energies)
        result_bins = get_bins(self.E_min, self.E_max, self.n_bins_result)
        sigma = self.experiment['res'](energies)
        bin_width = np.mean(np.diff(energies))
        events = np.array(smear_signal(rates, energies, sigma, bin_width))
        # re-bin final result to the desired number of bins
        events = self.chuck_integration(events, energies, result_bins)
        return events * self.experiment['exp_eff']

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
