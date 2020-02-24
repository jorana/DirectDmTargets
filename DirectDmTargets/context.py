from socket import getfqdn
import os

host = getfqdn()
print(f'Host: {host}')

if 'stbc' in host or 'nikhef' in host:
    context = {'software_dir': '/project/xenon/jorana/software/DD_DM_targets/',
               'results_dir': '/dcache/xenon/jorana/dddm/results/',
               'spectra_files': '/dcache/xenon/jorana/dddm/spectra/',
               'verne_folder': '/project/xenon/jorana/software/verne/',
               'verne_files': '/dcache/xenon/jorana/dddm/verne/'}
    if 'TMPDIR' in os.environ.keys():
        tmp_folder = os.environ['TMPDIR']
        print(f'found TMPDIR! on {host}')
    elif os.path.exists('/tmp/'):
        print("Setting tmp folder to /tmp/")
        tmp_folder = '/tmp/'
        if host == 'stbc-i1.nikhef.nl' or host == 'stbc-i2.nikhef.nl':
            # Fine, we can use the /tmp/ folder
            pass
        else:
            # Not fine, we cannot use the /tmp/ folder on the stoomboot nodes
            print(f'No tmp folder found on {host}. Envorionment vars:')
            for key in os.environ.keys():
                print(key)
            assert False                 
    assert os.path.exists(tmp_folder), f"Cannot find tmp folder at {tmp_folder}"
    context['tmp_folder'] = tmp_folder

elif host == 'DESKTOP-EC5OUSI.localdomain' or host == 'DESKTOP-URE1BBI.localdomain':
    context = {'software_dir': '/home/joran/google_drive/windows-anaconda/DD_DM_targets/',
               'results_dir': '/mnt/c/Users/Joran/dddm_data/results/',
               'spectra_files': '/mnt/c/Users/Joran/dddm_data/spectra/',
               'verne_folder': '/home/joran/google_drive/windows-anaconda/verne/',
               'verne_files': '/mnt/c/Users/Joran/dddm_data/verne/'}
    if os.path.exists('/tmp/'):
        print("Setting tmp folder to /tmp/")
        tmp_folder = '/tmp/'
    assert os.path.exists(tmp_folder), f"Cannot find tmp folder at {tmp_folder}"
    context['tmp_folder'] = tmp_folder
else:
    print(f'context.py::\tunknown host {host} be carefull here')
    # TODO
    context = {'software_dir': '../../DD_DM_targets/',
               'results_dir': '../../DD_DM_targets/data/results/',
               'spectra_files': '../../DD_DM_targets/data/results/spectra/',
               'verne_folder': '../../verne/',
               'verne_files': '../../verne/'}

    if os.path.exists('/tmp/'):
        print("Setting tmp folder to /tmp/")
        tmp_folder = '/tmp/'
    assert os.path.exists(tmp_folder), f"Cannot find tmp folder at {tmp_folder}"
    context['tmp_folder'] = tmp_folder
    for name in ['results_dir', 'spectra_files']:
        print(f'context.py::\tlooking for {name} in {context["name"]}')
        if not os.path.exists(context['name']):
            try:
                os.mkdir(context['name'])
            except:
                print(f'Could not find nor make {context["name"]}')
                raise OSError("Couldn't initialize folders correctly, please tailor context.py to your needs")

for key in context.keys():
    assert os.path.exists(context[key]), f'No folder at {context[key]}'
