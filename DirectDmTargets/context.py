import socket
from .utils import *

host = socket.getfqdn()
print(f'Host: {host}')

if 'stbc' in host or 'nikhef' in host:
    context = {'software_dir' : '/data/xenon/joranang/software/DD_DM_targets/',
               'results_dir' : '/dcache/xenon/jorana/dddm/results/',
               'specta_files' : '/dcache/xenon/jorana/dddm/spectra/',
               'verne_files' : '/dcache/xenon/jorana/dddm/verne/'}
elif 'local' in host:
    # TODO
    context = {'software_dir': '../',
               'results_dir': '/mnt/c/Users/Joran/dddm_data/results/',
               'specta_files': '/mnt/c/Users/Joran/dddm_data/spectra/',
               'verne_files': '/mnt/c/Users/Joran/dddm_data/verne/'
               }
else:
    # TODO
    context = {'software_dir' : '/data/xenon/joranang/software/DD_DM_targets/',
           'results_dir' : '/dcache/jorana/dddm/results/',
           'specta_files' : '/dcache/jorana/dddm/spectra/',
           'verne_files' : '/dcache/jorana/dddm/verne/'}