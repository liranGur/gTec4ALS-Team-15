import os
from typing import Tuple

import config
from config import DATA_DIR

import numpy as np
import pandas as pd
import scipy.io

if config.USE_FILE_PRINT:
    import fcntl


def load_data(file_names_dict_entry: Tuple, folder: str = DATA_DIR, as_df: bool = False):
    file_name = file_names_dict_entry[0]
    key = file_names_dict_entry[1]
    if as_df:
        return pd.DataFrame(scipy.io.loadmat(os.path.join(folder, file_name))[key])
    else:
        return np.array(scipy.io.loadmat(os.path.join(folder, file_name))[key])


def remove_trials(data: np.ndarray, labels: np.ndarray, to_remove: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param data: shape should be (trial, ............)
    :param labels: (trial,)
    :param to_remove: (trial,)
    :return:
    """
    if to_remove is None or len(to_remove) == 0:
        return data.copy(), labels.copy()
    idx_to_remove = np.where(to_remove == 1)
    return np.delete(data, idx_to_remove, axis=0), np.delete(labels, idx_to_remove)


def get_print_func(add_to_file_name: str = ""):
    if not config.USE_FILE_PRINT:
        return print

    add_to_file_name = add_to_file_name.replace("[", "").replace(" ", "_").replace("]", "").replace(",", "").replace("'","")
    print_file_name = f"my_out_{config.get_current_time_and_date()}_{add_to_file_name}.txt"
    print_file_path = os.path.join(config.OUT_DIR, print_file_name)
    os.makedirs(config.OUT_DIR, exist_ok=True)
    print_file = open(print_file_path, "w")
    print(f"Output file is: {print_file_path}")

    def worker(*args):
        print(args)
        fcntl.flock(print_file, fcntl.LOCK_EX)
        if isinstance(args, list):
            for arg in args:
                print_file.write(arg)
        else:
            print_file.write(str(args))
        print_file.write("\n")
        print_file.flush()
        fcntl.flock(print_file, fcntl.LOCK_UN)

    return worker


def get_all_configs():
    """
    This is just used to save all configs for printing
    :return: Dictionary with all configs
    """
    import inspect
    config_data = dir(config)
    ans = dict()
    for cand in config_data:
        obj = getattr(config, cand)
        if not inspect.isbuiltin(obj) and not inspect.isfunction(obj) and not inspect.ismodule(obj)\
                and not inspect.isclass(obj) and "__"not in cand:
            ans[cand] = obj
    return ans
