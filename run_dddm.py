import DirectDmTargets as dddm
import wimprates as wr
assert wr.__version__ !='0.2.2'
wr.__version__
import numpy as np
# import matplotlib.pyplot as plt
# import numericalunits as nu
# from tqdm import tqdm
# from scipy.integrate import quad as scipy_int
# import pandas as pd
# import scipy
import emcee
emcee.__version__
# import corner
import time
import argparse
# import os

# # Direct detection of Dark matter using different target materials #
# 
# Author:
# 
# Joran Angevaare <j.angevaare@nikef.nl>
# 
# Date:
# 
# 28 october 2019 
# 
print("run_dddm.py::\tstart")

parser = argparse.ArgumentParser(description="Running a fit for a certain set "
                                             "of parameters")
parser.add_argument('-mw', 
  type = np.float,
  default = 50.,
  help="wimp mass")
parser.add_argument('-cross_section', 
  type = np.float, 
  default = 1e-45, 
  help="wimp cross-section")
parser.add_argument('-poisson', 
  type = bool, 
  default = False, 
  help="Add poisson noise to the test dataset")
parser.add_argument('-nwalkers', 
  type = int, 
  default = 250, 
  help="walkers of MCMC")
parser.add_argument('-nsteps', 
  type = int, 
  default = 150, 
  help="steps of MCMC")
parser.add_argument('-notes', 
  type = str, 
  default = "default", 
  help="notes on particular settings")
args = parser.parse_args()

stats = dddm.MCMCStatModel("Xe")
stats.config['poisson'] = args.poisson
stats.config['notes'] = args.notes
stats.set_benchmark(mw=args.mw, sigma=args.cross_section)
stats.nwalkers = args.nwalkers
stats.nsteps = args.nsteps
print(f"run_dddm.py::\tstart for mw = {args.mw}, sigma = {args.cross_section}")
start = time.time()
stats.run_emcee()
end = time.time()
print(f"lasted {end-start} s = {(end-start)/3600} h")
stats.save_results()
assert stats.log['did_run']
print(f"run_dddm.py::\tfinished for mw = {args.mw}, "
      f"sigma = {args.cross_section}")

# ## Full dimensionality ##
print(f"run_dddm.py::\tfull fit")
print(f"run_dddm.py::\tstart for mw = {args.mw}, sigma = {args.cross_section}")
stats_full = dddm.MCMCStatModel("Xe")
stats_full.config['poisson'] = args.poisson
stats_full.config['notes'] = args.notes
stats_full.set_benchmark(mw=stats.config['mw'], sigma=stats.config['sigma'])
stats_full.nwalkers = stats.nwalkers
stats_full.nsteps = stats.nsteps * 2
stats_full.fit_parameters = stats_full.known_parameters
start = time.time()
stats_full.run_emcee()
end = time.time()
print(f"lasted {end-start} s = {(end-start)/3600} h")
stats_full.save_results()
assert stats.log['did_run']
print(f"run_dddm.py::\tfinished for mw = {args.mw}, "
      f"sigma = {args.cross_section}")
print("finished, bye bye")