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


def probabilities_decision_func(probs: np.ndarray, targets_diff: float, inner_diff: float, min_proba_strong: float,
                                min_proba_weak: float):
    """
    Select class based on probabilities
    :param probs: array of predict_proba result for model,
                  shape: #classes for classification, #2 (number of possible classifications)
    :param targets_diff: minimal difference between the top 2 probabilities for each possible class
    :param inner_diff:
    :param min_proba_strong:
    :param min_proba_weak:
    :return:
    """
    num_possible_classes = 2
    trigger_probs = probs[:, 1]
    top_indices = list(reversed(np.argsort(trigger_probs)[-num_possible_classes:]))
    top_vals = trigger_probs[top_indices]
    if top_vals[0] >= min_proba_strong and np.subtract(*top_vals) >= targets_diff / 2:
        return top_indices[0]
    if top_vals[0] >= min_proba_weak and np.subtract(*top_vals) >= targets_diff:
        return top_indices[0]
    if top_vals[0] > 1-inner_diff:
        return top_indices[0]

    return -1


def infer_data(folder_path: str, data_file_name: str, targets_diff: float = 0.1,
               inner_diff: float = 0.15, min_proba_strong: float = 0.5, min_proba_weak: float = 0.4) -> int:
    model = load_model(folder_path)
    log_data('received parameters', sys.argv[1:], f'{targets_diff=}, {inner_diff=}, {min_proba_strong=}, {min_proba_weak=}')
    data = get_inference_data(folder_path, data_file_name)
    preds = model.predict(data)
    log_data('Predictions are: ', preds.tolist())
    try:
        probs = model.predict_proba(data)
        log_data('predictions probabilities: {probs.tolist()}')
        if np.sum(preds) == 1:
            log_data('Using Preds !!!')
            class_ans = np.argwhere(preds == 1.)[0, 0]
        else:
            log_data('Preds failed trying probabilities')
            class_ans = probabilities_decision_func(probs, targets_diff, inner_diff, min_proba_strong, min_proba_weak)
            if class_ans == -1:
                log_data('Failed to predict based on probabilities')
                print(-1)
                exit(-1)

    except AttributeError:
        log_data('Model has no predict_proba using only preds')
        class_ans = np.argwhere(preds == 1.)[0, 0]
        if np.sum(preds) != 1:
            log_data('Failed to predict based on preds')
            exit(-2)

    matlab_class_idx = class_ans + 2  # 1 for matlab indexing and 1 for skipping idle class
    log_data(f'Selected class: {matlab_class_idx}')
    scipy.io.matlab.savemat(os.path.join(folder_path, 'predictions.mat'), {'predictions': matlab_class_idx})
    print(matlab_class_idx)
    return matlab_class_idx


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Not enough input parameters')
        exit(-1)
    start_log(False, 'infer')
    if len(sys.argv) == 7:
        selected_class_ = infer_data(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]),
                                     float(sys.argv[5]), float(sys.argv[6]))
    else:
        selected_class_ = infer_data(sys.argv[1], sys.argv[2])
