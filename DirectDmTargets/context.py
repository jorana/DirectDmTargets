"""Setup the file structure for the software. Specifies several folders:
software_dir: path of installation
"""

from socket import getfqdn
import os
from warnings import warn

# import DirectDmTargets
# import verne

host = getfqdn()
print(f'Host: {host}')

if 'stbc' in host or 'nikhef' in host:
    context = {'software_dir': '/project/xenon/jorana/software/DD_DM_targets/',
               'results_dir': '/data/xenon/joranang/dddm/results/',
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
            print(f'No tmp folder found on {host}. Environment vars:')
            for key in os.environ.keys():
                print(key)
            assert False
    assert os.path.exists(
        tmp_folder), f"Cannot find tmp folder at {tmp_folder}"
    context['tmp_folder'] = tmp_folder
    for key in context.keys():
        assert os.path.exists(context[key]), f'No folder at {context[key]}'

elif host == 'DESKTOP-EC5OUSI.localdomain' or host == 'DESKTOP-URE1BBI.localdomain':
    context = {
        'software_dir': '/home/joran/google_drive/windows-anaconda/DD_DM_targets/',
        'results_dir': '/mnt/c/Users/Joran/dddm_data/results/',
        'spectra_files': '/mnt/c/Users/Joran/dddm_data/spectra/',
        'verne_folder': '/home/joran/google_drive/windows-anaconda/verne/',
        'verne_files': '/mnt/c/Users/Joran/dddm_data/verne/'}
    if os.path.exists('/tmp/'):
        print("Setting tmp folder to /tmp/")
        tmp_folder = '/tmp/'
    elif 'TMPDIR' in os.environ.keys():
        tmp_folder = os.environ['TMPDIR']
        print(f'found TMPDIR! on {host}')
    elif 'TMP' in os.environ.keys():
        tmp_folder = os.environ['TMP']
        print(f'found TMPDIR! on {host}')
    assert os.path.exists(
        tmp_folder), f"Cannot find tmp folder at {tmp_folder}"
    context['tmp_folder'] = tmp_folder
    for key in context.keys():
        assert os.path.exists(context[key]), f'No folder at {context[key]}'
else:
    # Generally people will end up here
    print(f'context.py::\tunknown host {host} be careful here')
    # installation_folder = DirectDmTargets.__path__[0]
    installation_folder = os.path.abspath('./..')
    # vene_folder = verne.__path__[0]
    context = {
        'software_dir': installation_folder,
        'results_dir': os.path.join(
            installation_folder,
            'DD_DM_targets_data/'),
        'spectra_files': os.path.join(
            installation_folder,
            'DD_DM_targets_spectra/'),
        'verne_folder': os.path.join(
            installation_folder,
            '../verne/'),
        'verne_files': os.path.join(
            installation_folder,
            '../verne/'),
    }

    if os.path.exists('/tmp/'):
        print("Setting tmp folder to /tmp/")
        tmp_folder = '/tmp/'
    elif 'TMPDIR' in os.environ.keys():
        tmp_folder = os.environ['TMPDIR']
        print(f'found TMPDIR! on {host}')
    elif 'TMP' in os.environ.keys():
        tmp_folder = os.environ['TMP']
        print(f'found TMP! on {host}')
    assert os.path.exists(
        tmp_folder), f"Cannot find tmp folder at {tmp_folder}"
    context['tmp_folder'] = tmp_folder
    for name in ['results_dir', 'spectra_files']:
        print(f'context.py::\tlooking for {name} in {context}')
        if not os.path.exists(context[name]):
            try:
                os.mkdir(context[name])
            except Exception as e:
                warn(f'Could not find nor make {context[name]}')
                warn(f"Tailor context.py to your needs. Couldn't initialize "
                     f"folders correctly because of {e}.")
    for key in context.keys():
        if not os.path.exists(context[key]):
            warn(f'No folder at {context[key]}')

print(context)
