import json
import logging
import os
import random
import sys
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


def start_log(print_log):
    global PRINT_LOG
    logging.basicConfig(filename=os.path.join(sys.argv[1], 'py_debug.txt'), level=logging.DEBUG)
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
