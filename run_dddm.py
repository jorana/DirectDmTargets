import DirectDmTargets as dddm
import wimprates as wr
assert wr.__version__ !='0.2.2'
wr.__version__
import numpy as np
import matplotlib.pyplot as plt
import numericalunits as nu
from tqdm import tqdm
from scipy.integrate import quad as scipy_int
import pandas as pd
import scipy
import emcee
emcee.__version__
import corner
import time
import argparse
import os

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

parser = argparse.ArgumentParser(
    description="#TODO")
parser.add_argument('--mw', 
  type = np.float,
  default = 50.,
  help="wimp mass")
parser.add_argument('--cross_section', 
  type = np.float, 
  default = 1e-45, 
  help="arguments for the python script (specified by --target)")
args = parser.parse_args()

stats = dddm.MCMCStatModel("Xe")

stats.set_benchmark(mw=args.mw, 
	sigma=args.cross_section)

stats.nwalkers = 500
stats.nsteps = 500

print(f"run_dddm.py::\tstart for mw = {args.mw}, sigma = {args.cross_section}")
start = time.time()
stats.run_emcee()
end = time.time()
print(f"lasted {end-start} s = {(end-start)/3600} h")
stats.save_results()


assert stats.log['did_run']
print(f"run_dddm.py::\tfinished for mw = {args.mw}, sigma = {args.cross_section}")
# ## Full dimensionality ##
print(f"run_dddm.py::\tfull fit")
print(f"run_dddm.py::\tstart for mw = {args.mw}, sigma = {args.cross_section}")
stats_full = dddm.MCMCStatModel("Xe")
stats_full.set_benchmark(mw=stats.config['mw'], 
                         sigma=stats.config['sigma'])

stats_full.nwalkers = stats.nwalkers
stats_full.nsteps = stats.nsteps * 2

stats_full.fit_parameters = stats_full.known_parameters

start = time.time()
stats_full.run_emcee()
end = time.time()
print(f"lasted {end-start} s = {(end-start)/3600} h")
stats_full.save_results()


assert stats.log['did_run']
print(f"run_dddm.py::\tfinished for mw = {args.mw}, sigma = {args.cross_section}")
print("finished, bye bye")