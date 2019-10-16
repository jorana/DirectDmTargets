import numpy as np
import wimprates as wr
from scipy.integrate import quad as scipy_int


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


def dr_eff(E, sigma_res, dr_de=wr.rate_wimp_std,
           kwargs={'mw': 50, 'sigma_nucleon': 1e-45}):
    """Smears the WIMP spectrum with the detector resolution.

    :param E: energy (in GeV)
    :param sigma_res: function of E for calculating the detector resolution
    :param dr_de: function of the WIMP rate (unsmeared)
    :param kwargs: kwargs for ''dr_de''
    by default this is set to a 50 GeV wimp with coressection 10e-45

    returns: WIMP rate smeared by detector resolution
    """

    assert type(E) != np.ndarray, "Only single valued energies are allowed"

    def det_smear(E, Eprime):
        return (np.exp(-(E - Eprime) ** 2 / (2 * sigma_res(Eprime) ** 2))
                / (np.sqrt(2 * np.pi) * sigma_res(Eprime)))

    f = lambda Eprime: dr_de(Eprime, **kwargs) * det_smear(E, Eprime)

    # TODO: is this sufficient?
    E_max = 10 * E
    result, _ = scipy_int(f, 0, E_max)
    return result


# TODO I think that this function is bad (takes forever)
def N_r(E1, E2, e_eff, smearing=True, kwargs={'mw': 50, 'sigma_nucleon': 1e-45}):
    """Returns integral of given energy bin
    """
    if smearing:
        f = lambda E: dr_eff(E, det_res_Xe)
        res, _ = scipy_int(f, E1, E2)
    #         assert res/_ > 1000, "integrateion errors too big"

    else:
        f = lambda E: wr.rate_wimp_std(E, **kwargs)
        res, _ = scipy_int(f, E1, E2)
    #         assert res/_ > 1000, "integrateion errors too big"
    return e_eff * res

# def N_r(E1, E2, e_eff, kwargs = {'mw':50, 'sigma_nucleon':1e-45}):
#     f = lambda E: wr.rate_wimp_std(E, **kwargs)
#     res, _ = scipy_int(f, E1, E2)
#     assert res/_ > 1000, "integrateion errors too big"
#     return e_eff * res
