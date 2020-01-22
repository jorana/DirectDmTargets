"""Basic functions for saving et cetera"""

import numpy as np
import os
from datetime import datetime
from .context import *

def check_folder_for_file(file_path, max_iterations=30):
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
            # TODO
            if ".csv" in sub_dir:
                print("Error in this code, manually breaking but one should not end up here")
                break
            this_dir = base_dir + "/" + sub_dir
            if not os.path.exists(this_dir):
                print(f'check_save_folder::\tmaking {this_dir}')
                try:
                    os.mkdir(this_dir)
                except FileExistsError:
                    print("This is strange. We got a FileExistsError for a path to be made, maybe another instance has created this path too")
                    pass
            base_dir = this_dir
        assert_str = f'check_save_folder::\tsomething failed. Cannot find {last_folder}'

        if not os.path.exists(last_folder):
            print(file_path)
            print(base_dir)
            print(max_iterations)
            print(last_folder)
            assert False, assert_str


def now():
    '''

    :return: datetime.datetime string with day, hour, minutes
    '''
    return datetime.now().isoformat(timespec='minutes')


def load_folder_from_context(request):
    '''

    :param request: request a named path from the context
    :return: the path that is requested
    '''
    try:
        folder = context[request]
    except KeyError:
        print(f'load_folder_from_context::\tRequesting {request} but that is not in {context.keys()}')
        raise KeyError
    if not os.path.exists(folder):
        raise FileNotFoundError(f'load_folder_from_context::\tCould not find {folder}')
    # Should end up here:
    return folder


def get_result_folder(*args):
    '''
    bridge to work with old code when context was not yet implemented
    '''
    if args:
        print(f'get_result_folder::\tfunctionallity depcricated ignoring {args}')
    print(f'get_result_folder::\trequested folder is {context["results_dir"]}')
    return load_folder_from_context('results_dir')


def get_verne_folder():
    '''
    bridge to work with old code when context was not yet implemented
    '''
    return load_folder_from_context('verne_files')


def is_savable_type(item):
    '''

    :param item: input of any type.
    :return: bool if the type is saveable by checking if it is in a limitative list
    '''
    if type(item) in [list, np.array, np.ndarray, int, str, np.int, np.float,
                      bool, np.float64]:
        return True
    return False


def convert_dic_to_savable(config):
    '''

    :param config: some dictionary to save
    :return: string-like object that should be savable.
    '''
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
    '''

    :param save_dir: requested name of folder to open in the result folder
    :param force_index: option to force to write to a number (must be an override!)
    :return: the name of the folder as was saveable (usually input + some number)
    '''
    base = get_result_folder()
    save = save_dir
    files = os.listdir(base)
    files = [f for f in files if save in f]
    if not save + '0' in files and not force_index:
        # First file in the results folder with this name
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
    # this is where we going to save
    save_dir = base + save + str(index) + '/'
    print('open_save_dir::\tusing ' + save_dir)
    if force_index is False:
        assert not os.path.exists(save_dir), 'Trying to override another directory, this would be very messy'
        os.mkdir(save_dir)
    else:
        assert os.path.exists(save_dir), "specify existing directory, exit"
        for file in os.listdir(save_dir):
            print('open_save_dir::\tremoving ' + save_dir + file)
            os.remove(save_dir + file)
    return save_dir
