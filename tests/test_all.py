import os
from sys import platform

print(f'Running on {platform}')

if 'win' in platform:
    os.system("python run_dddm_emcee.py -nwalkers 10 -nsteps 10")
    os.system("python run_dddm_multinest.py -nlive 10 -tol 0.9 -sampler nestle -shielding no")
else:
    print("happy travis")
    os.system("python run_dddm_emcee.py -nwalkers 10 -nsteps 10")
    os.system("python run_dddm_multinest.py -nlive 10 -tol 0.999 -sampler nestle -shielding no")
    os.system("python run_dddm_multinest.py -nlive 10 -tol 0.999 -shielding no -save_intermediate yes")

    os.system("python run_dddm_multinest.py -nlive 10 -tol 0.9999 -shielding yes")
    os.system("mpiexec -n 3 python run_dddm_multinest.py -nlive 10 -tol 0.999 -shielding no -save_intermediate no -multicore_hash TESTHASH")