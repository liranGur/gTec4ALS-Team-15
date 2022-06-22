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
    return x_data[:3]


def probabilities_decision_func(probs, targets_diff=0.1, inner_diff=0.15, min_proba_strong=0.5, min_proba_weak=0.4):
    num_possible_classes = 2
    trigger_probs = probs[:, 1]
    top_indices = list(reversed(np.argsort(trigger_probs)[-num_possible_classes:]))
    top_vals = trigger_probs[top_indices]
    if top_vals[0] >= min_proba_weak and np.subtract(*top_vals) >= targets_diff / 2:
        return top_indices[0]
    if top_vals[0] >= min_proba_strong:
        if np.subtract(*top_vals) >= targets_diff:
            return top_indices[0]
    if probs[0, top_indices[0]] - probs[1, top_indices[0]] > inner_diff:
        return top_indices[0]

    return -1


def infer_data(folder_path: str, data_file_name: str, targets_diff: float = 0.1,
               inner_diff: float = 0.15, min_proba_strong: float = 0.5, min_proba_weak: float = 0.4) -> int:
    model = load_model(folder_path)
    log_data('received parameters', sys.argv[1:], f'  thresholds: {min_prob} , min-df: {min_dif}')
    data = get_inference_data(folder_path, data_file_name)
    try:
        preds_probs = model.predict_proba(data)
        log_data('predictions probabilities: ', preds_probs.tolist())
        preds = model.predict(data)
        log_data('Predictions are: ', preds.tolist())
        if max(preds_probs) < min_prob:
            log_data('max value is too small')
            exit(-1)
        top_idx = np.argsort(preds_probs)[-2:]
        if top_idx[0] - top_idx[1] < min_dif:
            log_data('Difference between top 2 classes is too small')
            exit(-2)
        matlab_class_idx = top_idx[0] + 2   # 1 for matlab indexing and 1 for skipping idle class
    except AttributeError:
        log_data("Can't use predict proba")
        preds = model.predict(data)
        log_data('Predictions are: ', preds.tolist())
        if np.sum(preds) != 1:
            log_data('Failed to ge a class sum != 1')
            exit(-3)
        matlab_class_idx = np.argwhere(preds == 1.)[0, 0] + 2  # 1 for matlab indexing and 1 for skipping idle class
    log_data(f'Selected class: {matlab_class_idx}')
    scipy.io.matlab.savemat(os.path.join(folder_path, 'predictions.mat'), {'predictions': matlab_class_idx})
    print(matlab_class_idx)
    return matlab_class_idx


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Not enough input parameters')
        exit(-1)
    start_log(False, 'infer')
    selected_class_ = infer_data(*sys.argv[1:])
