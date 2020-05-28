"""Introduce detector effects into the expected detection spectrum"""

import numba
import numpy as np
import pandas as pd
from warnings import warn
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
    # For CDMS-lite
    # From https://arxiv.org/pdf/1808.09098.pdf (table II.)
    """
    :return: detector resolution for Ge detector
    """
    sigma_e = 10. / 1e3  # eV to keV
    param_a = 5.0e-3  # dimensionless
    param_b = 0.85 / 1e3  # eV to keV
    # TODO does this give the right dimensions?
    return np.sqrt(sigma_e ** 2 +
                   param_b * E +
                   (param_a * E) ** 2)


def det_res_superCDMS(E):
    resolution = 10  # eV
    resolution /= 1e3  # keV
    # https://arxiv.org/abs/1610.00006
    return np.full(len(E), resolution)


def det_res_XENON1T(E):
    warn('Deprecation warning: use det_res_XENON1T_update')
    return det_res_Xe(E)


def det_res_XENON1T_update(E):
    """
    Detector resolution of XENON1T. See caption figure 6 https://arxiv.org/abs/2003.03825
    :param E: energy in keV
    :return: resolution at E
    """
    a = 31.71
    b = 0.15
    sigma_over_E_percent = b + a / np.sqrt(E)
    return E * sigma_over_E_percent / 100


def det_res_DarkSide(E):
    warn('Warning use det_res_XENON1T_update() instead')
    return det_res_Ar(E)


def migdal_background_XENON1T(e_min, e_max, nbins):
    """
    :return: background for Xe detector
    """
    bin_width = (e_max - e_min) / nbins  # keV
    # Assume that:
    #   A) The BG is 10x lower than in https://www.nature.com/articles/s41586-019-1124-4
    #   B) The BG is flat
    bg_rate = 80 / 10  # 1/(keV * t * yr)
    # Assume flat background over entire energy range
    # True to first order below 200 keV

    if e_min > e_max or e_max > 200:
        raise ValueError(f'Assume flat background only below 200 keV ({e_min}, {e_max})')
    return np.full(nbins, bg_rate) * bin_width


@numba.jit(nopython=True)
def migdal_background_CDMS(e_min, e_max, nbins):
    """
    :return: background for Ge detector
    """
    bin_width = (e_max - e_min) / nbins  # keV
    bins = np.linspace(e_min, e_max, nbins)
    res = []
    conv_units = 1.0e-3 * (1 / 365.25)
    for i in range(nbins):
        res.append(
            CDMS_background_functions(bins[i]) * conv_units)

    return np.array(res) * bin_width


def migdal_background_superCDMS_Ge_HV(e_min, e_max, nbins):
    """
    :return: background for Ge HV detector
    """
    # https://arxiv.org/abs/1610.00006
    # Assume flat bg from 32Si (Fig. 4 & Table V), ignore other isotopes.
    bg_rate = 27  # counts/kg/keV/year
    conv_units = 1.0e3  # Tonne
    bin_width = (e_max - e_min) / nbins  # keV
    if e_min > e_max or e_max > 20:
        raise ValueError(f'Assume flat background only below 20 keV ({e_min}, {e_max})')
    return np.full(nbins, bg_rate*conv_units) * bin_width


def migdal_background_superCDMS_Si_HV(e_min, e_max, nbins):
    """
    :param E: recoil energy (in keV)
    :return: background for Si HV detector
    """
    # https://arxiv.org/abs/1610.00006
    # Assume flat bg from 32Si (Fig. 4 & Table V), ignore other isotopes.
    bg_rate = 300  # counts/kg/keV/year
    conv_units = 1.0e3  # Tonne
    bin_width = (e_max - e_min) / nbins  # keV
    if not e_max < 20:  # 20 keV
            raise ValueError(f'Assume flat background only below 100 keV ({e_min}, {e_max})')
    return np.full(nbins, bg_rate*conv_units) * bin_width


def migdal_background_superCDMS_Ge_iZIP(e_min, e_max, nbins):
    """
    :return: background for Ge iZIP detector
    """
    bin_width = (e_max - e_min) / nbins # keV
    bg_rate = 370  # counts/kg/keV/year
    conv_units = 1.0e3  # Tonne
    bin_width = (e_max - e_min) / nbins  # keV
    if not e_max < 20:  # 20 keV
        raise ValueError(f'Assume flat background only below 100 keV ({e_min}, {e_max})')
    return np.full(nbins, bg_rate*conv_units) * bin_width


def migdal_background_superCDMS_Si_iZIP(e_min, e_max, nbins):
    """
    :param E: recoil energy (in keV)
    :return: background for Si iZIP detector
    """
    # https://arxiv.org/abs/1610.00006
    # Assume flat bg from 3H (Fig. 4 & Table V), ignore other isotopes.
    bg_rate = 370  # counts/kg/keV/year
    conv_units = 1.0e3  # Tonne
    bin_width = (e_max - e_min) / nbins  # keV
    if not e_max < 100:
          raise ValueError(f'Assume flat background only below 100 keV ({e_min}, {e_max})')
    return np.full(nbins, bg_rate*conv_units) * bin_width


# TODO
#  NOTE WE ASSUME FOR BOTH HV AND IZIP THE SAME VALUE BUT ONLY IZIP IS GIVEN
def nr_background_superCDMS_Ge(e_min, e_max, nbins):
    """
    :return: background for Ge iZIP/HV detector
    """
    # https://arxiv.org/abs/1610.00006
    # Assume flat bg from 3H (Fig. 4 & Table V), ignore other isotopes.
    bg_rate = 3300*1e-6  # counts/kg/keV/year
    conv_units = 1.0e3  # Tonne
    bin_width = (e_max - e_min) / nbins  # keV

    # Assume only flat over first 20 keV thereafter negligible.
    energies = np.linspace(e_min, e_max, nbins)
    res = np.zeros(nbins)
    res[energies < 20] = bg_rate*conv_units
    return res * bin_width


# TODO
#  NOTE WE ASSUME FOR BOTH HV AND IZIP THE SAME VALUE BUT ONLY IZIP IS GIVEN
def nr_background_superCDMS_Si(e_min, e_max, nbins):
    """
    :param E: recoil energy (in keV)
    :return: background for Si iZIP/HV detector
    """
    # https://arxiv.org/abs/1610.00006
    # Assume flat bg from 3H (Fig. 4 & Table V), ignore other isotopes.
    bg_rate = 2900*1e-6  # counts/kg/keV/year
    conv_units = 1.0e3  # Tonne
    bin_width = (e_max - e_min) / nbins  # keV

    # Assume only flat over first 20 keV thereafter negligible.
    energies = np.linspace(e_min, e_max, nbins)
    res = np.zeros(nbins)
    res[energies < 20] = bg_rate * conv_units
    return res * bin_width


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


# Set the default benchmark for a 50 GeV WIMP with a cross-section of 1e-45 cm^2
benchmark = {'mw': 50., 'sigma_nucleon': 1e-45}

# Set up a dictionary of the different detectors
# Each experiment below lists:
# Name :{Interaction type (type0, exposure [ton x yr] (exp.), cut efficiency (cut_eff),
# nuclear recoil acceptance (nr_eff), energy threshold [keV] (E_thr), resolution function (res)
experiment = {
    'Xe': {'material': 'Xe', 'type': 'SI', 'exp': 5., 'cut_eff': 0.8, 'nr_eff': 0.5, 'E_thr': 10.,
            'location': "XENON",
           'res': det_res_Xe},
    'Ge': {'material': 'Ge', 'type': 'SI', 'exp': 3., 'cut_eff': 0.8, 'nr_eff': 0.9, 'E_thr': 10.,
            'location': "SUF",
           'res': det_res_Ge},
    'Ar': {'material': 'Ar', 'type': 'SI', 'exp': 10., 'cut_eff': 0.8, 'nr_eff': 0.8, 'E_thr': 30.,
            'location': "XENON",
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
        'bg_func': migdal_background_XENON1T,
    },
    'Ge_migd': {
        'material': 'Ge',
        'type': 'migdal',
        # TODO 100 kg yr (for 56 + 44 Ge in Table I.)
        'exp': 100 * 1.e-3,
        # https://www.slac.stanford.edu/exp/cdms/ScienceResults/Publications/PhysRevD.95.082002.pdf
        'cut_eff': 0.8,
        'nr_eff': 0.9,
        'E_thr': 70. / 1e3,  # Assume similar to CDMSlite https://arxiv.org/pdf/1808.09098.pdf
        "location": "SUF",
        'res': det_res_CDMS,
        'bg_func': migdal_background_CDMS,
    },
    'Ar_migd': {
        'material': 'Ar',
        'type': 'migdal',
        'exp': 10. * 5,
        'cut_eff': 0.8,
        'nr_eff': 0.8,
        # TODO otherwise no results at all, but they really don't have a 5 keV threshold
        'E_thr': 3.,
        "location": "XENON",
        'res': det_res_DarkSide,
        'bg_func': migdal_background_Darkside,
    },
    'Xe_migd_bg': {
        'material': 'Xe',
        'type': 'migdal_bg',
        'exp': 5 * 5,  # aim for 2025 (5 yr * 5 ton)
        'cut_eff': 0.8,
        # TODO
        #  'nr_eff': 1,  # Not required for ER-type??
        'nr_eff': 0.5,
        'E_thr': 1.0,  # assume slightly lower than https://arxiv.org/abs/1907.12771
        'location': "XENON",
        'res': det_res_XENON1T,
        'bg_func': migdal_background_XENON1T,
    },
    # https://arxiv.org/abs/1610.00006
    'Ge_migd_iZIP_bg': {
        'material': 'Ge',
        'type': 'migdal_bg',
        'exp': 56 * 1.e-3,  # Tonne year
        'cut_eff': 0.75,  # p. 11, right column
        'nr_eff': 0.5,  # p. 11, left column NOTE: migdal is ER type!
        'E_thr': 350. / 1e3,  # table VIII, Eph
        "location": "SNOLAB",
        'res': det_res_superCDMS,
        'bg_func': migdal_background_superCDMS_Ge_iZIP,
    },
     'Ge_migd_iZIP_Si_bg': {
        'material': 'Si',
        'type': 'migdal_bg',
        'exp': 4.8 * 1.e-3,  # Tonne year
        'cut_eff': 0.75,  # p. 11, right column
        'nr_eff': 0.675,  # p. 11, left column NOTE: migdal is ER type!
        'E_thr': 175./1e3,  # table VIII, Eph
        "location": "SNOLAB",
        'res': det_res_superCDMS,
        'bg_func':  migdal_background_superCDMS_Si_iZIP,
    },
    'Ge_migd_HV_bg': {
        'material': 'Ge',
        'type': 'migdal_bg',
        'exp': 44 * 1.e-3,  # Tonne year
        'cut_eff': 0.85,  # p. 11, right column
        'nr_eff': 0.5,  # p. 11, left column NOTE: migdal is ER type!
        'E_thr':  100. / 1e3,  # table VIII, Eph
        "location": "SNOLAB",
        'res': det_res_superCDMS,  # TODO
        'bg_func': migdal_background_superCDMS_Ge_HV,
    },
     'Ge_migd_HV_Si_bg': {
        'material': 'Si',
        'type': 'migdal_bg',
        'exp': 9.6 * 1.e-3,  # Tonne year
        # https://www.slac.stanford.edu/exp/cdms/ScienceResults/Publications/PhysRevD.95.082002.pdf
        'cut_eff': 0.85,  # p. 11, right column
        'nr_eff': 0.675,  # p. 11, left column NOTE: migdal is ER type!
        'E_thr':  100. / 1e3,  # table VIII, Eph
        "location": "SNOLAB",
        'res': det_res_superCDMS,
        'bg_func': migdal_background_superCDMS_Si_HV,
    },
    'Xe_migd_tmp_bg': {
        'material': 'Xe',
        'type': 'migdal_bg',
        'exp': 5 * 5,  # aim for 2025 (5 yr * 5 ton)
        'cut_eff': 0.8,
        # TODO
        #  'nr_eff': 1,  # Not required for ER-type??
        'nr_eff': 0.5,
        'E_thr': 1.0,  # assume slightly lower than https://arxiv.org/abs/1907.12771
        'location': "XENON",
        'res': det_res_XENON1T_update,
        'bg_func': migdal_background_XENON1T,
    },
    }

# And calculate the effective exposure for each
for name in experiment.keys():
    experiment[name]['exp_eff'] = (experiment[name]['exp'] *
                                   experiment[name]['cut_eff'] *
                                   experiment[name]['nr_eff'])
    experiment[name]['name'] = name

for det in ['Xe_migd_tmp_bg', 'Ge_migd_HV_Si_bg', 'Ge_migd_HV_bg',
            'Ge_migd_iZIP_Si_bg', 'Ge_migd_iZIP_bg']:
    migd_exp = experiment[det].copy()
    migd_exp['type'] = 'SI_bg'
    if 'Ge_' in det:
        migd_exp['E_thr'] = {'Ge_migd_HV_bg': 40. / 1e3,  # table VIII, Enr
                             'Ge_migd_HV_Si_bg': 78. / 1e3,  # table VIII, Enr
                             'Ge_migd_iZIP_bg': 272. / 1e3,  # table VIII, Enr
                             'Ge_migd_iZIP_Si_bg': 166. / 1e3,  # table VIII, Enr
                             }[det]
        migd_exp['nr_eff'] = 0.85  # p. 11, left column
        migd_exp['bg_func'] = {'Ge_migd_HV_bg': nr_background_superCDMS_Ge,
                               'Ge_migd_iZIP_bg': nr_background_superCDMS_Ge,
                               'Ge_migd_HV_Si_bg': nr_background_superCDMS_Si,
                               'Ge_migd_iZIP_Si_bg': nr_background_superCDMS_Si}[det]
    name = det.replace('_migd', '')
    if name in experiment:
        raise ValueError(f'{name} already in {experiment.keys()}')
    else:
        experiment.update({name: migd_exp})


# Make a copy with setting background to True!
exp_names = experiment.keys()
for name in list(exp_names):
    bg_name = name + '_bg'
    if bg_name not in exp_names:
        experiment[bg_name] = experiment[name].copy()
        experiment[bg_name]['type'] = experiment[bg_name]['type'] + '_bg'


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
            # see formula (5) in https://arxiv.org/abs/1012.3458
            res = res + (bin_width * rate[j] *
                         (1 / (np.sqrt(2 * np.pi) * sigma[j])) *
                         np.exp(-((energy[i] - energy[j]) ** 2 / (2 * sigma[j] ** 2)))
                         )
            # TODO
            #  # at the end of the spectrum the bg-rate drops as the convolution does
            #  # not take into account the higher energies.
            #  weight = length / (j-length)
            #  res = res * weight
        result.append(res)
    return np.array(result)


class DetectorSpectrum(GenSpectrum):
    def __init__(self, *args):
        GenSpectrum.__init__(self, *args)
        # GenSpectrum generates a number of bins (default 10), however, since a
        # numerical integration is performed in compute_detected_spectrum, this
        # number is multiplied here.
        self.rebin_factor = 10
        self.n_bins_result = None
        # Please not that this is NOT pretty. It was a monkey patch implemented since
        # many spectra were already computed using this naming hence we have to deal
        # with this lack of clarity in earlier coding in this manner.
        self.add_background = True if 'bg' in self.experiment['type'] else False

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
        # get the spectrum
        rates = self.spectrum_simple([self.mw, self.sigma_nucleon])
        # if this option is set to true, add a background component
        if self.add_background:
            # pay close attention, the events in the bg_func are already taking into
            # account the det. efficiency et cetera. Hence the number here should be
            # multiplied by the total exposure (rather than the effective exposure that
            # is multiplied by at the end of this subroutine. Hence the bg rates obtained
            # from that function is multiplied by the ratio between the two.
            rates += self.experiment['bg_func'](self.E_min,
                                                self.E_max,
                                                self.n_bins) * (
                    self.experiment['exp'] / self.experiment['exp_eff'])
        energies = self.get_bin_centers()

        # Set the rate to zero for energies smaller than the threshold
        rates = self.above_threshold(rates, energies)
        result_bins = get_bins(self.E_min, self.E_max, self.n_bins_result)
        sigma = self.experiment['res'](energies)
        bin_width = np.mean(np.diff(energies))
        # Smear (using numerical integration) the rates with the detector resolution
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
        result = self.set_negative_to_zero(result)

        return result
