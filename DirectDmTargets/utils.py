"""Basic functions for saving et cetera"""

import numpy as np
import os
from datetime import datetime
from .context import *

def check_folder_for_file(file_path, max_iterations=30, verbose = 1):
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
            assert_str = f"check_folder_for_file::\tstarting from a folder " \
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
                if verbose:
                    print(f'check_folder_for_file::\tmaking {this_dir}')
                try:
                    os.mkdir(this_dir)
                except FileExistsError:
                    print("This is strange. We got a FileExistsError for a path to be made, maybe another instance has created this path too")
                    pass
            base_dir = this_dir
        assert_str = f'check_folder_for_file::\tsomething failed. Cannot find {last_folder}'

        if not os.path.exists(last_folder):
            print(file_path)
            print(base_dir)
            print(max_iterations)
            print(last_folder)
            assert False, assert_str


def now(tstart = None):
    '''

    :return: datetime.datetime string with day, hour, minutes
    '''
    res = datetime.now().isoformat(timespec='minutes')
    if tstart:
        res+= f'\tdt=\t{(datetime.now()-tstart).seconds} s'
    return res


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


def open_save_dir(save_dir, base=None, force_index=False, hash=None):
    '''

    :param save_dir: requested name of folder to open in the result folder
    :param base: folder where the save dir is to be saved in. This is the results folder by default
    :param force_index: option to force to write to a number (must be an override!)
    :return: the name of the folder as was saveable (usually input + some number)
    '''
    if base is None:
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
    if hash:
        assert force_index is False, f'do not set hash to {hash} and force_index to {force_index} simultaneously'
        save_dir = base + save + '_HASH'+ str(hash) + '/'
        if not os.path.exists(save_dir):
            try:
                os.mkdir(save_dir)
            except FileExistsError:
                # starting up on multiple cores causes the command above to be executed simultaniously
                pass
        else:
            files_in_dir = os.listdir(save_dir)
            if len(files_in_dir):
                print(f'WARNING writing to {save_dir}. There are files in this dir: {files_in_dir} ')
        print('open_save_dir::\tusing ' + save_dir)
        return save_dir
    if force_index is False:
        assert not os.path.exists(save_dir), 'Trying to override another directory, this would be very messy'
        os.mkdir(save_dir)
    else:
        if not os.path.exists(save_dir): #, "specify existing directory, exit"
            os.mkdir(save_dir)
        else:
            for file in os.listdir(save_dir):
                print('open_save_dir::\tremoving ' + save_dir + file)
                os.remove(save_dir + file)
    print('open_save_dir::\tusing ' + save_dir)
    return save_dir


# TODO comment
def str_in_list(string, _list):
    '''checks if sting is in any of the items in _list
    if so return that item'''
    for name in _list:
        if string in name:
            return name
    raise FileNotFoundError(f'No name named {string} in {_list}')


def is_str_in_list(string, _list, verbose =1):
    '''checks if sting is in any of the items in _list.
    :return bool:'''
    print(f'looking for {string} in {_list}')
    for name in _list:
        if string in name:
            if verbose: print(f'{string} is in  {name}!')
            return True
        else:
            if verbose: print(f'{string} is not in  {name}')
    return False

def add_identifier_to_safe(name, verbose = 1):
    '''
    :param name: takes name
    :return: abs_file_name, exist_csv
    '''

    assert '.csv' in name, f"{name} is not .csv"
    # where to look
    csv_path = '/'.join(name.split('/')[:-1]) + '/'
    # what to look for
    csv_key = name.split('/')[-1].strip('.csv')
    # What can we see
    if not os.path.exists(csv_path):
        exist_csv = False
        if not host in name:
            abs_file_name = name.replace('.csv', f'-H{host}-P{os.getpid()}.csv')
        else:
            abs_file_name = name
        return exist_csv, abs_file_name

    files_in_folder = os.listdir(csv_path + '/')
    if verbose:
        print(f'VerneSHM::\tlooking for "{csv_key}" in "{csv_path}". '
          f'That folder has "{files_in_folder}". '
          f'\n\tDoes it have the right file?\n\t{is_str_in_list(csv_key, files_in_folder)}')

    if is_str_in_list(csv_key, files_in_folder):
        if verbose:
            print(f'Using {str_in_list(csv_key, files_in_folder)} since it has {csv_key}')
        exist_csv = True
        abs_file_name = csv_path + str_in_list(csv_key, files_in_folder)
        print(f'Using {abs_file_name} as input')

    else:
        exist_csv = False
        if not host in name:
            abs_file_name = name.replace('.csv', f'-{host}.csv')
        else:
            abs_file_name = name

    return exist_csv, abs_file_name

    # elif str_in_list(csv_key, files_in_folder):
    # print(f'Using {str_in_list(csv_key, files_in_folder)} since it has {csv_key}')
    # file_name = csv_path + str_in_list(csv_key, files_in_folder)
    # print(f'Using {file_name} for the velocity distribution')

    return abs_file_name, exist_csv

