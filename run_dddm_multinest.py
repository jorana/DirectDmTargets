import DirectDmTargets as dddm
import wimprates as wr
import numpy as np
import argparse
assert wr.__version__ != '0.2.2'
import random
import multiprocessing
import time
import os
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

#TODO:
verbose = True

parser = argparse.ArgumentParser(description="Running a fit for a certain set "
                                             "of parameters")
parser.add_argument('-mw',
                    type=np.float,
                    default=50.,
                    help="wimp mass")
parser.add_argument('-cross_section',
                    type=np.float,
                    default=-45,
                    help="wimp cross-section")
parser.add_argument('-poisson',
                    type=bool,
                    default=False,
                    help="Add poisson noise to the test dataset")
parser.add_argument('-nlive',
                    type=int,
                    default=1024,
                    help="live points used by nestle")
parser.add_argument('-tol',
                    type=float,
                    default=0.1,
                    help="tolerance for opimization (see nestle option dlogz)")
parser.add_argument('-notes',
                    type=str,
                    default="default",
                    help="notes on particular settings")
parser.add_argument('-bins',
                    type=int,
                    default=10,
                    help="the number of energy bins")
parser.add_argument('-target',
                    type=str,
                    default='Xe',
                    help="Target material of the detector (Xe/Ge/Ar)")
parser.add_argument('-nparams',
                    type=int,
                    default=2,
                    help="Number of parameters to fit")
parser.add_argument('-priors_from',
                    type=str,
                    default="Pato_2010",
                    help="Obtain priors from paper <priors_from>")
parser.add_argument('-verbose',
                    type=float,
                    default=0,
                    help="Set to 0 (no print statements), 1 (some print statements) or >1 (a lot of print statements). Set the level of print statements while fitting.")
parser.add_argument('-shielding',
                    type=str,
                    default="default",
                    help="yes / no / default, override internal determination if we need to take into account earth shielding.")
parser.add_argument('-save_intermediate',
                    type=str,
                    default="yes",
                    help="yes / no / default, override internal determination if we need to take into account earth shielding.")
parser.add_argument('-multicore_hash',
                    type=str,
                    default="",
                    help="no / default, override internal determination if we need to take into account earth shielding.")

args = parser.parse_args()
yes_or_no = {"yes" : True, "no" : False}

print(f"info\nn_cores:{multiprocessing.cpu_count}\npid{os.getpid()}")
time.sleep(5)

print(f"run_dddm_nestle.py::\tstart for mw = {args.mw}, sigma = "
      f"{args.cross_section}. Fitting {args.nparams} parameters")
stats = dddm.NestedSamplerStatModel(args.target, args.verbose)
stats.sampler = 'multinest'
# stats.sampler = 'nestle'
if args.shielding != "default":
    stats.config['earth_shielding'] = yes_or_no[args.shielding.lower()]
    stats.set_models()
else:
    assert False
#TODO
stats.config['save_intermediate'] = yes_or_no[args.save_intermediate.lower()]
stats.config['poisson'] = args.poisson
stats.config['notes'] = args.notes
stats.config['n_energy_bins'] = args.bins
stats.set_prior(args.priors_from)
#TODO change to set_fit_parameters
stats.fit_parameters = stats.known_parameters[:args.nparams]
stats.set_benchmark(mw=args.mw, sigma=args.cross_section)
stats.eval_benchmark()
stats.nlive = args.nlive
stats.config['nlive']= args.nlive
stats.tol = args.tol
# stats.run_nestle()
if args.multicore_hash != "":
    stats.get_save_dir(hash= args.multicore_hash)
    stats.get_tmp_dir(hash = args.multicore_hash)
if stats.sampler == 'multinest':
    stats.run_multinest()
stats.save_results()
assert stats.log['did_run']
stats.empty_garbage()

print(f"run_dddm_nestle.py::\tfinished for mw = {args.mw}, "
      f"sigma = {args.cross_section}")
print("finished, bye bye")
