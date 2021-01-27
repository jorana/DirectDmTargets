import os
from sys import platform

print(f'Running on {platform}')

if 'win' in platform:
    def test_emcee_windows():
        os.system("python run_dddm_emcee.py "
                  "-nwalkers 10 "
                  "-nsteps 10")

    def test_multinest_windows():
        os.system("python run_dddm_multinest.py "
                  "-nlive 10 "
                  "-tol 0.9 "
                  "-sampler nestle "
                  "-shielding no")
else:
    print("happy travis")

    def test_emcee():
        os.system("python run_dddm_emcee.py "
                  "-nwalkers 10 "
                  "-nsteps 10")

    def test_multinest_no_shield():
        os.system("python run_dddm_multinest.py "
                  "-nlive 10 "
                  "-tol 0.999 "
                  "-sampler nestle "
                  "-shielding no")

    def test_multinest_no_shield_save_interim():
        os.system("python run_dddm_multinest.py "
                  "-nlive 10 "
                  "-tol 0.999 "
                  "-shielding no "
                  "-save_intermediate yes")

    def test_multinest_shielding():
        os.system("python run_dddm_multinest.py "
                  "-nlive 10 "
                  "-tol 0.9999 "
                  "-shielding yes")

    def test_multinest_multicore():
        os.system("mpiexec -n 3 python run_dddm_multinest.py "
                  "-nlive 10 "
                  "-tol 0.999 "
                  "-shielding no "
                  "-save_intermediate no "
                  "-multicore_hash TESTHASH")