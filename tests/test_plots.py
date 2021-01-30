import DirectDmTargets


def test_ll_s():
    DirectDmTargets.plot_basics.plt_ll_sigma_spec(bins=2)

def test_ll_m():
    DirectDmTargets.plot_basics.plt_ll_mass_spec(bins=3)

def test_plt_b():
    DirectDmTargets.plot_basics.plt_priors(itot=10)
