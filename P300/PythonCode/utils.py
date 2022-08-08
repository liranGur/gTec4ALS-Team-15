import json
import logging
import os
import random
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import scipy.io
from config import Const


random.seed(42)
np.random.seed(42)
PRINT_LOG = True


def load_mat_data(file_name_dict_entry: Tuple, folder: str, as_df: bool = False):
    file_name = file_name_dict_entry[0]
    key = file_name_dict_entry[1]
    if as_df:
        return pd.DataFrame(scipy.io.loadmat(os.path.join(folder, file_name))[key])
    else:
        return np.array(scipy.io.loadmat(os.path.join(folder, file_name))[key])


def load_parameters(folder: str) -> Dict:
    raw_parameters = load_mat_data(Const.training_parameters, folder, as_df=False)
    parameters_names = raw_parameters[0].dtype.names
    parameters = dict()
    for idx, name in enumerate(parameters_names):
        parameters[name] = raw_parameters[0][0][idx].ravel()[0]
    return parameters


def start_log(print_log, name: str, log_folder: str):
    global PRINT_LOG
    logging.basicConfig(filename=os.path.join(log_folder, f'py_debug_{name}.txt'), level=logging.DEBUG)
    PRINT_LOG = print_log


def log_data(*args):
    global PRINT_LOG
    for arg in args:
        if PRINT_LOG:
            print(arg)
        if isinstance(arg, str):
            logging.debug(arg)
        else:
            if isinstance(arg, set):
                arg = list(arg)
            logging.debug(json.dumps(arg))


def convert_triggers_times_to_sample_idx(triggers_times: np.ndarray, hz: int):
    start_times = np.expand_dims(triggers_times[:, 0], axis=1)
    real_times = triggers_times[:, 1:] - start_times
    samples = real_times * hz
    return np.round(samples).astype(int)


def convert_sample_idx_to_trigger_time(samples: np.ndarray, trigger_sample_idx: int, hz: int):
    if trigger_sample_idx == -1 or trigger_sample_idx == 0:
        return np.arange(len(samples)) / hz
    else:
        post_times = np.arange(len(samples) - trigger_sample_idx) / hz
        pre_times = np.flip(np.arange(-1, - (len(samples) - len(post_times) + 1), step=-1) / hz)
        times = np.concatenate((pre_times, post_times))
        return times
