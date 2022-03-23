import os
from typing import Tuple
import numpy as np
import pandas as pd
import scipy.io


def load_mat_data(file_name_dict_entry: Tuple, folder: str, as_df: bool = False):
    file_name = file_name_dict_entry[0]
    key = file_name_dict_entry[1]
    if as_df:
        return pd.DataFrame(scipy.io.loadmat(os.path.join(folder, file_name))[key])
    else:
        return np.array(scipy.io.loadmat(os.path.join(folder, file_name))[key])
