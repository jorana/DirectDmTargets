# DD_DM_targets
[![CodeFactor](https://www.codefactor.io/repository/github/jorana/directdmtargets/badge)](https://www.codefactor.io/repository/github/jorana/directdmtargets)
[![Pytest](https://github.com/jorana/DirectDmTargets/workflows/Pytest/badge.svg)](https://github.com/jorana/DirectDmTargets/actions?query=workflow%3APytest)
[![Coverage Status](https://coveralls.io/repos/github/jorana/DirectDmTargets/badge.svg?branch=master)](https://coveralls.io/github/jorana/DirectDmTargets?branch=master)

Probing the complementarity of several targets used in Direct Detection Experiments for Dark Matter

# Author
Joran Angevaare <j.angevaare@nikhef.nl>

# Requirements
 - [Wimprates](https://github.com/jorana/wimprates).
 - [verne](https://github.com/jorana/verne)
 - Optimizer:
    - [multinest](https://github.com/JohannesBuchner/PyMultiNest)
    - [emcee](https://emcee.readthedocs.io/en/stable/)
    - [nestle](http://kylebarbary.com/nestle/)

# Usage

# Options
 - Multiprocessing
 - Earth shielding
 - Computing cluster utilization

# Installation (linux)
```bash
echo 'Quick installing in conda env'
pip install pymultinest
yes | conda install -c conda-forge multinest
yes | conda install -c anaconda mpi4py
yes | conda install -c conda-forge emcee
pip install git+https://github.com/jorana/wimprates
pip install git+https://github.com/jorana/verne
```

