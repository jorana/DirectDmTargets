import DirectDmTargets as dddm
import wimprates as wr
import numpy as np
import numericalunits as nu
import matplotlib.pyplot as plt


def test_simple_spectrum():
    energies = np.linspace(0.01, 20, 50)

    # dr/dr
    dr = ((nu.keV * (1000 * nu.kg) * nu.year) *
          wr.rate_migdal(energies * nu.keV,
                         mw=5 * nu.GeV / nu.c0 ** 2,
                         sigma_nucleon=1e-35 * nu.cm ** 2))

    plt.plot(energies, dr, label="WIMPrates SHM")
    dr = ((nu.keV * (1000 * nu.kg) * nu.year) *
          wr.rate_migdal(energies * nu.keV,
                         mw=0.5 * nu.GeV / nu.c0 ** 2,
                         sigma_nucleon=1e-35 * nu.cm ** 2))

    plt.plot(energies, dr, label="WIMPrates SHM")

    plt.xlabel("Recoil energy [keV]")
    plt.ylabel("Rate [events per (keV ton year)]")

    plt.xlim(0, energies.max())
    plt.yscale("log")

    plt.ylim(1e-4, 1e8)
    plt.clf()
    plt.close()


def _detector_spectrum_inner(use_SHM):
    det = 'Xe'
    mw = 1
    sigma = 1e-35
    nbins = 10
    E_max = None
    events = dddm.GenSpectrum(mw, sigma, use_SHM, dddm.experiment[det])
    events.n_bins = nbins
    if E_max:
        events.E_max = E_max
    events.get_data(poisson=False)


def test_detector_spectrum():
    use_SHM = dddm.SHM()
    _detector_spectrum_inner(use_SHM)


def test_shielded_detector_spectrum():
    use_SHM = dddm.VerneSHM()
    _detector_spectrum_inner(use_SHM)
