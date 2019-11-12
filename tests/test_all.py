import os

print("happy travis")
os.system("python run_dddm_emcee.py -nwalkers 10 -nsteps 10")
os.system("python run_dddm_nestle.py -nlive 10 -tol 0.9")