import json
import os
import sys
import joblib
import numpy as np
import scipy.io.matlab

from utils import load_mat_data, log_data, start_log
from config import const
from ModelSearchDataUtils import get_manipulation_func, final_eeg_to_train_data


def load_model(folder_path: str):
    model_file = list(filter(lambda file_name: 'model' in file_name, os.listdir(folder_path)))[0]
    model = joblib.load(os.path.join(folder_path, model_file))
    return model


def get_inference_data(folder_path: str, data_file_name: str) -> np.ndarray:
    with open(os.path.join(folder_path, const.search_result_file), 'r') as file:
        model_search_res = json.load(file)
    channels = model_search_res['channels']
    manipulation_func = get_manipulation_func(model_search_res['mode'])
    processed_eeg = load_mat_data((data_file_name, const.processed_eeg[1]), folder_path)
    manipulated_data = manipulation_func([processed_eeg[:, :, channels, :]])
    x_data, _ = final_eeg_to_train_data(np.array(manipulated_data[0]), None)
    return x_data


def infer_data(folder_path: str, data_file_name: str) -> int:
    model = load_model(folder_path)
    data = get_inference_data(folder_path, data_file_name)
    preds = model.predict(data)
    if np.sum(preds) != 1:
        log_data(f"Predictions aren't good enough - preds: {preds}", )
        exit(-1)
    log_data(f'predicitons: {preds}')
    selected_class = np.argwhere(preds == 1.)[0, 0] + 2     # TODO think how to add here skip idle class data
    log_data(f'Selected class: {selected_class}')
    scipy.io.matlab.savemat(os.path.join(folder_path, 'predictions.mat'), {'predictions': selected_class})
    print(selected_class)
    return selected_class


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Not enough input parameters')
        exit(-1)
    start_log(False, 'infer')
    selected_class_ = infer_data(*sys.argv[1:])
