"""Basic functions for saving et cetera"""

import numpy as np
import os
from datetime import datetime


def check_folder_for_file(file_path, max_iterations=10):
    '''

    :param file_path: path with one or more subfolders
    :param max_iterations: max number of lower lying subfolders
    '''
    last_folder = "/".join(file_path.split("/")[:-1])
    max_iterations = np.min([max_iterations, len(file_path.split("/"))-1])
    if os.path.exists(last_folder):
        # Folder does exist. No need do anything.
        return
    else:
        
        if file_path[0] == '/':
            base_dir = '/' + file_path.split("/")[1]
            start_i = 2
        else:
            base_dir = file_path.split("/")[0]
            start_i = 1
            assert_str = f"check_save_folder::\tstarting from a folder " \
                         f"({base_dir}) that cannot be found"
            assert os.path.exists(base_dir), assert_str
        # Start from 1 (since that is basedir) go until second to last since that is the file name
        for sub_dir in file_path.split("/")[start_i:max_iterations]:
            this_dir = base_dir + "/" + sub_dir
            if not os.path.exists(this_dir):
                print(f'check_save_folder::\tmaking {this_dir}')
                os.mkdir(this_dir)
            base_dir = this_dir
        assert_str = f'check_save_folder::\tsomething failed. Cannot find {last_folder}'

        assert os.path.exists(last_folder), assert_str


def now():
    '''

    :return: datetime.datetime string with day, hour, minutes
    '''
    return datetime.now().isoformat(timespec='minutes')


def get_result_folder(current_folder='.'):
    folder = 'results/'
    if not os.path.exists(folder):
        folder = '../' + folder
    if not os.path.exists(folder):
        raise FileNotFoundError(f'Could not find {folder}')
    return folder
    # TODO
    # for i in range(10):
    #     if os.path.exists(current_folder + folder):
    #         return current_folder + folder
    #     else:
    #         folder = '../' + folder
    # raise FileNotFoundError(f'No folder was found between {current_folder} and {folder}')


# TODO UGLY
def get_verne_folder():
    folder = '../../verne/'
    if not os.path.exists(folder):
        folder = '../verne/'
    if not os.path.exists(folder):
        raise FileNotFoundError(f'Could not find {folder}')
    return folder


if not os.path.exists(get_verne_folder()):
    raise FileNotFoundError(f"no folder at {get_verne_folder}")

# if not os.path.exists(get_result_folder()):
#     os.mkdir(get_result_folder())


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
