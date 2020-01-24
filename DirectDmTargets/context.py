from socket import getfqdn
import os
# from .utils import *

host = getfqdn()
print(f'Host: {host}')

if 'stbc' in host or 'nikhef' in host:
    context = {'software_dir': '/data/xenon/joranang/software/DD_DM_targets/',
               'results_dir': '/dcache/xenon/jorana/dddm/results/',
               'specta_files': '/dcache/xenon/jorana/dddm/spectra/',
               'verne_folder': '/data/xenon/joranang/software/verne/',
               'verne_files': '/dcache/xenon/jorana/dddm/verne/'}
elif host == 'DESKTOP-EC5OUSI.localdomain':
    context = {'software_dir': '/home/joran/google_drive/windows-anaconda/DD_DM_targets/',
               'results_dir': '/mnt/c/Users/Joran/dddm_data/results/',
               'specta_files': '/mnt/c/Users/Joran/dddm_data/spectra/',
               'verne_folder': '/home/joran/google_drive/windows-anaconda/verne/',
               'verne_files': '/mnt/c/Users/Joran/dddm_data/verne/'}
else:
    print(f'context.py::\tunknown host {host} be carefull here')
    # TODO
    context = {'software_dir': '../../DD_DM_targets/',
               'results_dir': '../../DD_DM_targets/data/results/',
               'specta_files': '../../DD_DM_targets/data/results/spectra/',
               'verne_folder': '../../verne/',
               'verne_files': '../../verne/'}
    for name in ['results_dir', 'specta_files']:
        print(f'context.py::\tlooking for {name} in {context["name"]}')
        if not os.path.exists(context['name']):
            try:
                os.mkdir(context['name'])
            except:
                print(f'Could not find nor make {context["name"]}')
                raise OSError("Couldn't initialize folders correctly, please tailor context.py to your needs")

for key in context.keys():
    assert os.path.exists(context[key]), f'No folder at {context[key]}'
