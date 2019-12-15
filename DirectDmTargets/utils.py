"""Basic functions for saving et cetera"""

import numpy as np
import os


def get_result_folder():
    return 'results/'

#TODO UGLY
def get_verne_folder():
    return '../../verne/'

if not os.path.exists(get_verne_folder()):
    raise FileNotFoundError(f"no folder at {get_verne_folder}")

if not os.path.exists(get_result_folder()):
    os.mkdir(get_result_folder())


def is_savable_type(item):
    if type(item) in [list, np.array, np.ndarray, int, str, np.int, np.float,
                      bool, np.float64]:
        return True
    return False


def convert_dic_to_savable(config):
    result = config.copy()
    for key in result.keys():
        if is_savable_type(result[key]):
            pass
        elif type(result[key]) == dict:
            result[key] = convert_dic_to_savable(result[key])
        else:
            result[key] = str(result[key])
    return result


def open_save_dir(save_dir, force_index=False):
    base = get_result_folder()
    save = save_dir
    files = os.listdir(base)
    files = [f for f in files if save in f]
    if not save + '0' in files and not force_index:
        index = 0
    elif force_index is False:
        index = 0
        for f in files:
            try:
                index = max(int(f.split(save)[-1]) + 1, index)
            except ValueError:
                # this means that f.split(save)[-1] is not an integer, thus,
                # that folder uses a different naming convention and we can
                # ignore it.
                pass
    else:
        index = force_index
    save_dir = base + save + str(index) + '/'
    print('open_save_dir::\tusing ' + save_dir)
    if force_index is False:
        os.mkdir(save_dir)
    else:
        assert os.path.exists(save_dir), "specify existing directory, exit"
        for file in os.listdir(save_dir):
            print('open_save_dir::\tremoving ' + save_dir + file)
            os.remove(save_dir + file)
    return save_dir

