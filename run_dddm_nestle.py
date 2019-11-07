import DirectDmTargets as dddm
import wimprates as wr
assert wr.__version__ !='0.2.2'
import numpy as np
import time
import argparse

# # Direct detection of Dark matter using different target materials #
# 
# Author:
# 
# Joran Angevaare <j.angevaare@nikef.nl>
# 
# Date:
# 
# 7 november 2019 
# 
print("run_dddm_nestle.py::\tstart")

parser = argparse.ArgumentParser(description="Running a fit for a certain set "
                                             "of parameters")
parser.add_argument('-mw', 
  type = np.float,
  default = 50.,
  help="wimp mass")
parser.add_argument('-cross_section', 
  type = np.float, 
  default = -45, 
  help="wimp cross-section")
parser.add_argument('-poisson', 
  type = bool, 
  default = False, 
  help="Add poisson noise to the test dataset")
# parser.add_argument('-nwalkers', 
#   type = int, 
#   default = 250, 
#   help="walkers of MCMC")
# parser.add_argument('-nsteps', 
#   type = int, 
#   default = 150, 
#   help="steps of MCMC")
parser.add_argument('-nlive', 
  type = int, 
  default = 1024, 
  help="live points used by nestle")
parser.add_argument('-tol', 
  type = float, 
  default = 0.1, 
  help="tolerance for opimization (see nestle option dlogz)")
parser.add_argument('-notes', 
  type = str, 
  default = "default", 
  help="notes on particular settings")
args = parser.parse_args()

print(f"run_dddm_nestle.py::\tstart for mw = {args.mw}, sigma = {args.cross_section}")
stats = dddm.NestleStatModel("Xe")
stats.config['poisson'] = args.poisson
stats.config['notes'] = args.notes
stats.set_benchmark(mw=args.mw, sigma=args.cross_section)
stats.nlive = args.nlive
stats.tol = args.tol
stats.run_nestle()
stats.save_results()
assert stats.log['did_run']
print(f"run_dddm.py::\tfinished for mw = {args.mw}, "
      f"sigma = {args.cross_section}")

# ## Full dimensionality ##
print(f"run_dddm_nestle.py::\tfull fit")
print(f"run_dddm_nestle.py::\tstart for mw = {args.mw}, sigma = {args.cross_section}")
stats_full = dddm.NestleStatModel("Xe")
stats_full.config['poisson'] = args.poisson
stats_full.config['notes'] = args.notes
stats_full.set_benchmark(mw=args.mw, sigma=args.cross_section)
stats_full.nlive = args.nlive
stats_full.tol = args.tol
stats_full.fit_parameters = stats_full.known_parameters
stats_full.run_nestle()
stats_full.save_results()
assert stats.log['did_run']
print(f"run_dddm_nestle.py::\tfinished for mw = {args.mw}, "
      f"sigma = {args.cross_section}")
print("finished, bye bye")