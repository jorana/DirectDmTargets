import DirectDmTargets


def test_utils():
    l = [str(i) for i in range(10)]
    assert DirectDmTargets.utils.is_str_in_list('1', l)
    assert DirectDmTargets.utils.str_in_list('1', l)
