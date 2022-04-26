import os
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import scipy.io

from config import Const


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
