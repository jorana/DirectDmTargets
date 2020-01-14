import socket
from .utils import *

host = socket.getfqdn()
print(f'Host: {host}')

if 'stbc' in host:
    context = {'software_dir' : '/data/xenon/joranang/software/DD_DM_targets/',
               'results_dir' : '/dcache/xenon/jorana/dddm/results/',
               'specta_files' : '/dcache/xenon/jorana/dddm/spectra/',
               'verne_files' : '/dcache/xenon/jorana/dddm/verne/'}
elif 'local' in host:
    # TODO
    context = {'software_dir' : '../',
               'results_dir' : get_results_folder() '/dcache/jorana/dddm/results/',
               'specta_files' : get_results_folder()  + '/spectra/',
               'verne_files' : get_verne_folder()}
else:
    # TODO
    context = {'software_dir' : '/data/xenon/joranang/software/DD_DM_targets/',
           'results_dir' : '/dcache/jorana/dddm/results/',
           'specta_files' : '/dcache/jorana/dddm/spectra/',
           'verne_files' : '/dcache/jorana/dddm/verne/'}