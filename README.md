# DD_DM_targets
[![Build Status](https://travis-ci.com/jorana/DD_DM_targets.svg?token=2MSppqzrkto9C3uuoWiK&branch=master)](https://travis-ci.com/jorana/DirectDmTargets)

Probing the complementarity of several targets used in Direct Detection Experiments for Dark Matter

Install wimprates from source (since pip gives a too low version):
<https://github.com/jorana/wimprates.git> (originally <https://github.com/JelleAalbers/wimprates/tree/master/wimprates>)

Also install multihist from source
https://github.com/JelleAalbers/multihist.git

Install verne from source:
<https://github.com/jorana/verne> originally from <https://github.com/bradkav/verne>

# Installation
 - Works both on Windows and Linux. To install:
 
``git clone https://github.com/jorana/wimprates``

``pip install -e wimprates``

``git clone https://github.com/jorana/verne.git``

``git clone https://github.com/JohannesBuchner/PyMultiNest``

``pip install -e PyMultiNest``

``conda install -c conda-forge multinest``

``conda install -c conda-forge/label/cf201901 multinest ``

``git clone https://github.com/jorana/DD_DM_targets.git``

``cd DD_DM_targets``

``pip install -r requirements.txt``

``python setup.py install``






# You may also do something like:
 - make a new conda env:

``conda create -n dddm numpy scipy git python=3.7 matplotlib pandas ipython jupyter numba``

``conda activate DD_DM``

``conda install -c conda-forge emcee`` 

``pip install corner numericalunits``

--``pip install numericalunits``--

``pip install jupyter_contrib_nbextensions``

``jupyter contrib nbextension install`` (--user may be needed)

``jupyter nbextension enable spellchecker/main``
