import DirectDmTargets as dddm
import numpy as np
from hypothesis import given, settings, strategies


@settings(deadline=None, max_examples=10)
@given(strategies.floats(0, 10),
       strategies.floats(5, 20),
       strategies.integers(2, 10),
       )
def test_cdms_migdal_bg(emin, emax, nbins):
    res = dddm.migdal_background_CDMS(emin, emax, nbins)
    assert len(res) == nbins
    if nbins:
        assert np.all(res > 0)
