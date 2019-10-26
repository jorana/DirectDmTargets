# DD_DM_targets
Probing the complementarity of several targets used in Direct Detection Experiments for Dark Matter

Install wimprates from source (since pip gives a too low version):
<https://github.com/JelleAalbers/wimprates/tree/master/wimprates>

## Requirements ##
 - make a new conda env:

``conda create -n DD_DM numpy scipy git python=3.7 matplotlib pandas ipython jupyter numba corner``

``conda activate DD_DM``

``conda install -c conda-forge emcee`` 

``pip install wimprates``

``pip install numericalunits``

``pip install jupyter_contrib_nbextensions``

``jupyter contrib nbextension install`` (--user may be needed)

``jupyter nbextension enable spellchecker/main``


