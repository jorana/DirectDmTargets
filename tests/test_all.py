import os

os.system("python ../run_dddm.py -nwalkers 5 -nsteps 7")
os.system("python ../run_dddm_nestle.py -live 10 -tol 0.9")