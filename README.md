# DD_DM_targets
[![CodeFactor](https://www.codefactor.io/repository/github/jorana/directdmtargets/badge)](https://www.codefactor.io/repository/github/jorana/directdmtargets)
[![Pytest](https://github.com/jorana/DirectDmTargets/workflows/Pytest/badge.svg)](https://github.com/jorana/DirectDmTargets/actions?query=workflow%3APytest)
[![Coverage Status](https://coveralls.io/repos/github/jorana/DirectDmTargets/badge.svg?branch=restructure_dddm)](https://coveralls.io/github/jorana/DirectDmTargets?branch=restructure_dddm)

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
conda install -c conda-forge/label/cf202003 multinest
conda install -c anaconda mpi4py
conda install -c conda-forge emcee
pip install git+https://github.com/jorana/wimprates
pip install git+https://github.com/jorana/verne
pip install git+https://github.com/JohannesBuchner/PyMultiNest
```

