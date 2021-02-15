"""Basic functions for saving et cetera"""

from DirectDmTargets import context
import numpy as np
import os
import datetime
import uuid


def check_folder_for_file(file_path):
    """
    :param file_path: path with one or more subfolders
    """
    last_folder = os.path.split(file_path)[0]
    os.makedirs(last_folder, exist_ok=True)

    if not os.path.exists(last_folder):
        raise OSError(f'Could not make {last_folder} for saving {file_path}')


def now(tstart=None):
    """

    :return: datetime.datetime string with day, hour, minutes
    """
    res = datetime.datetime.now().isoformat(timespec='minutes')
    if tstart:
        res += f'\tdt=\t{(datetime.datetime.now() - tstart).seconds} s'
    return res


def load_folder_from_context(request):
    """

    :param request: request a named path from the context
    :return: the path that is requested
    """
    try:
        folder = context.context[request]
    except KeyError:
        print(
            f'load_folder_from_context::\tRequesting {request} but that is not in {context.context.keys()}')
        raise KeyError
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f'load_folder_from_context::\tCould not find {folder}')
    # Should end up here:
    return folder


def get_result_folder(*args):
    """
    bridge to work with old code when context was not yet implemented
    """
    if args:
        print(f'get_result_folder::\tfunctionality deprecated ignoring {args}')
    print(
        f'get_result_folder::\trequested folder is {context.context["results_dir"]}')
    return load_folder_from_context('results_dir')


def get_verne_folder():
    """
    bridge to work with old code when context was not yet implemented
    """
    return load_folder_from_context('verne_files')


def is_savable_type(item):
    """

    :param item: input of any type.
    :return: bool if the type is saveable by checking if it is in a limitative list
    """
    savables = (list, np.ndarray, int, str, np.int, np.float, bool, np.float64)
    if isinstance(item, savables):
        return True
    return False


def convert_dic_to_savable(config):
    """

    :param config: some dictionary to save
    :return: string-like object that should be savable.
    """
    result = config.copy()
    for key in result.keys():
        if is_savable_type(result[key]):
            pass
        elif isinstance(result[key], dict):
            result[key] = convert_dic_to_savable(result[key])
        else:
            result[key] = str(result[key])
    return result


def open_save_dir(save_dir, base=None, force_index=False, _hash=None):
    """

    :param save_dir: requested name of folder to open in the result folder
    :param base: folder where the save dir is to be saved in. This is the results folder by default
    :param force_index: option to force to write to a number (must be an override!)
    :param _hash: add a has to save dir to avoid duplicate naming conventions while running multiple jobs
    :return: the name of the folder as was saveable (usually input + some number)
    """
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
    save_dir = os.path.join(base, save + str(index))
    if _hash:
        assert force_index is False, f'do not set _hash to {_hash} and force_index to {force_index} simultaneously'
        save_dir = os.path.join(base, (save + '_HASH' + str(_hash)))
        if not os.path.exists(save_dir):
            try:
                os.mkdir(save_dir)
            except FileExistsError:
                # starting up on multiple cores causes the command above to be
                # executed simultaneously
                pass
        else:
            files_in_dir = os.listdir(save_dir)
            if len(files_in_dir):
                print(
                    f'WARNING writing to {save_dir}. There are files in this dir: {files_in_dir} ')
        print('open_save_dir::\tusing ' + save_dir)
        return save_dir
    if force_index is False:
        assert not os.path.exists(
            save_dir), 'Trying to override another directory, this would be very messy'
        os.mkdir(save_dir)
    else:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        else:
            for file in os.listdir(save_dir):
                print('open_save_dir::\tremoving ' + save_dir + file)
                os.remove(save_dir + file)
    print('open_save_dir::\tusing ' + save_dir)
    return save_dir


def str_in_list(string, _list):
    """checks if sting is in any of the items in _list
    if so return that item"""
    for name in _list:
        if string in name:
            return name
    raise FileNotFoundError(f'No name named {string} in {_list}')


def is_str_in_list(string, _list, verbose=0):
    """checks if sting is in any of the items in _list.
    :return bool:"""
    if len(_list) < 10:
        print(f'is_str_in_list::\tlooking for {string} in {_list}')
    for name in _list:
        if string in name:
            if verbose:
                print(f'is_str_in_list::\t{string} is in  {name}!')
            return True
        if verbose:
            print(f'is_str_in_list::\t{string} is not in  {name}')
    return False


def add_identifier_to_safe(name, verbose=1):
    """
    :param name: takes name
    :param verbose: print level
    :return: abs_file_name, exist_csv
    """

    assert '.csv' in name, f"{name} is not .csv"
    # where to look
    csv_path = name.replace('.csv', "")
    # what to look for
    csv_key = os.path.split(name)[-1].replace('.csv', "")

    if os.path.exists(name) and not os.stat(name).st_size:
        # Check that the file we are looking for is not an empty file, that
        # would be bad.
        print(f"WARNING:\t removing empty file {name}")
        os.remove(name)

    # What can we see
    if not os.path.exists(csv_path):
        exist_csv = False
        if context.host not in name:
            abs_file_name = name.replace(
                '.csv', f'-H{context.host}-P{os.getpid()}.csv')
        else:
            abs_file_name = name
        return exist_csv, abs_file_name

    files_in_folder = os.listdir(csv_path)
    if verbose:
        print(
            f'VerneSHM::\tlooking for "{csv_key}" in "{csv_path}".\n\tDoes it have the'
            f' right file?\n\t{is_str_in_list(csv_key, files_in_folder)}')
        if len(files_in_folder) < 5:
            print(f'That folder has "{files_in_folder}". ')
    if is_str_in_list(csv_key, files_in_folder):
        if verbose:
            print(
                f'VerneSHM::\tUsing {str_in_list(csv_key, files_in_folder)} since it has {csv_key}')
        exist_csv = True
        abs_file_name = csv_path + str_in_list(csv_key, files_in_folder)
        print(f'VerneSHM::\tUsing {abs_file_name} as input')
    else:
        print("VerneSHM::\tNo file found")
        exist_csv = False
        if context.host not in name:
            abs_file_name = name.replace(
                '.csv', f'-H{context.host}-P{os.getpid()}.csv')
        else:
            abs_file_name = name

    return exist_csv, os.path.abspath(abs_file_name)

    # elif str_in_list(csv_key, files_in_folder):
    # print(f'Using {str_in_list(csv_key, files_in_folder)} since it has {csv_key}')
    # file_name = csv_path + str_in_list(csv_key, files_in_folder)
    # print(f'Using {file_name} for the velocity distribution')

    # return abs_file_name, exist_csv


def unique_hash():
    return uuid.uuid4().hex[15:]


def remove_nan(x, maskable=False):
    """
    :param x: float or array
    :param maskable: array to take into consideration when removing NaN and/or
    inf from x
    :return: x where x is well defined (not NaN or inf)
    """
    if not isinstance(maskable, bool):
        assert_string = f"match length maskable ({len(maskable)}) to length array ({len(x)})"
        assert len(x) == len(maskable), assert_string
    if maskable is False:
        mask = ~not_nan_inf(x)
        return masking(x, mask)
    return masking(x, ~not_nan_inf(maskable) ^ not_nan_inf(x))


def not_nan_inf(x):
    """
    :param x: float or array
    :return: array of True and/or False indicating if x is nan/inf
    """
    if np.shape(x) == () and x is None:
        x = np.nan
    try:
        return np.isnan(x) ^ np.isinf(x)
    except TypeError:
        return np.array([not_nan_inf(xi) for xi in x])


def masking(x, mask):
    """
    :param x: float or array
    :param mask: array of True and/or False
    :return: x[mask]
    """
    assert len(x) == len(
        mask), f"match length mask {len(mask)} to length array {len(x)}"
    try:
        return x[mask]
    except TypeError:
        return np.array([x[i] for i in range(len(x)) if mask[i]])


def bin_edges(a, b, n):
    """
    :param a: lower limit
    :param b: upper limit
    :param n: number of bins
    :return: bin edges for n bins

    """
    _, edges = np.histogram(np.linspace(a, b), bins=n)
    return edges


def get_bins(a, b, n):
    """
    :param a: lower limit
    :param b: upper limit
    :param n: number of bins
    :return: center of bins
    """
    result = np.vstack((bin_edges(a, b, n)[0:-1], bin_edges(a, b, n)[1:]))
    return np.transpose(result)
