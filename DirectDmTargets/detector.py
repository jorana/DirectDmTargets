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


# Set the default benchmark for a 50 GeV WIMP with a cross-section of 1e-45 cm^2
benchmark = {'mw': 50, 'sigma_nucleon': 1e-45}

# Set up a dictionary of the different detectors
detectors = {
    'Xe': {'exp': 5, 'cut_eff': 0.8, 'nr_eff': 0.5, 'E_thr': 10,
           'res': det_res_Xe},
    'Ge': {'exp': 3, 'cut_eff': 0.8, 'nr_eff': 0.9, 'E_thr': 10,
           'res': det_res_Ge},
    'Ar': {'exp': 10, 'cut_eff': 0.8, 'nr_eff': 0.8, 'E_thr': 30,
           'res': det_res_Ar}
    }
# And calculate the effective exposure for each
for name in detectors.keys():
    detectors[name]['exp_eff'] = (detectors[name]['exp'] *
                                  detectors[name]['cut_eff'] *
                                  detectors[name]['nr_eff'])
    detectors[name]['name'] = name
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
        rates[energies < self.detector['E_thr']] = 0
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
        energies = self.get_bin_centers()
        rates = self.above_threshold(rates, energies)
        result_bins = get_bins(self.E_min, self.E_max, self.n_bins_result)
        sigma = self.detector['res'](energies)
        bin_width = np.mean(np.diff(energies))
        events = np.array(smear_signal(rates, energies, sigma, bin_width))
        # re-bin final result to the desired number of bins
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
