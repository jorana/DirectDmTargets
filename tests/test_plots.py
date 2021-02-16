import DirectDmTargets
import matplotlib.pyplot as plt
import numpy as np

def test_ll_s():
    DirectDmTargets.plot_basics.plt_ll_sigma_spec(bins=2)
    plt.clf()
    plt.close()


def test_ll_m():
    DirectDmTargets.plot_basics.plt_ll_mass_spec(bins=3)
    plt.clf()
    plt.close()


def test_plt_b():
    DirectDmTargets.plot_basics.plt_priors(itot=10)
    plt.clf()
    plt.close()


def test_ll_function():
    DirectDmTargets.plot_basics.show_ll_function(20)
    plt.clf()
    plt.close()


def test_simple_hist():
    DirectDmTargets.plot_basics.simple_hist(np.linspace(0, 3, 3))
    plt.clf()
    plt.close()
