# DD_DM_targets
[![Build Status](https://travis-ci.com/jorana/DD_DM_targets.svg?token=2MSppqzrkto9C3uuoWiK&branch=master)](https://travis-ci.com/jorana/DirectDmTargets)
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
pip install git+https://github.com/jorana/wimprates
```

