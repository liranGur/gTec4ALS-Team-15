import itertools
import json
import logging
import os
import random
import sys
from functools import partial
from typing import List, Dict, Union, Tuple, Callable, Set, Any, Optional
import numpy as np
import shutil
from p_tqdm import p_map
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import joblib
from datetime import datetime

from utils import load_mat_data
from config import Const


random.seed(42)
np.random.seed(42)


def log_data(*args):
    for arg in args:
        print(arg)
        if isinstance(arg, str):
            logging.debug(arg)
        else:
            if isinstance(arg, set):
                arg = list(arg)
            logging.debug(json.dumps(arg))


def load_data(folder_path):
    log_data(f'loading data from: {folder_path}')
    processed_eeg = load_mat_data(Const.processed_eeg, folder_path)
    training_labels = load_mat_data(Const.training_labels, folder_path)
    return processed_eeg, training_labels


def final_eeg_to_train_data(eeg_data: np.ndarray, labels: Union[List, np.ndarray],
                            remove_idle_cls: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts the final eeg data (after manipulation on channels to final model train data)
    :param eeg_data: eeg data after all pre processing, shape: #trials, #classes, sample
    :param labels: list with target values. size: #trials
    :param remove_idle_cls: True if there is an idle class
    :return: Tuple of:
        * train data - shape:#trial*#non_idle_classes, sample
        * training labels - vector of 0 or 1 for binary classification, shape: #trial*#non_idle_classes, sample
    """
    log_data(f'skipping idle class: {remove_idle_cls}')
    labels = np.array(labels).ravel() - 1       # Convert to python indexing
    num_trials = eeg_data.shape[0]
    if remove_idle_cls:
        # remove idle class from data
        eeg_data = eeg_data[:, 1:, :]
        # This is done because the target values with idle class are from 2 to number of classes and here the indices
        # are 0 to num of classes, so we need to fix by 2 (1 for matlab index and 1 for skipping idle class)
        labels = labels - 1
    num_classes = eeg_data.shape[1]
    x_train = np.reshape(eeg_data, (num_trials * num_classes, eeg_data.shape[2]))
    y_train = np.zeros(x_train.shape[0])
    for trial_idx, curr_trial_label in enumerate(labels):
        for cls in range(num_classes):
            y_train[trial_idx*num_classes + cls] = int(curr_trial_label == cls)

    return x_train, y_train


def select_best_model(results: List[Dict]) -> Dict:
    """
    Some heuristics to select a single model to use
    :param results: list of dictionaries of type result search. They must have accuracy, and channel
    :return: dictionary of the final best result
    """
    # filter by max accuracy
    max_acc = max(res['accuracy'] for res in results)
    filtered_results = list(filter(lambda res: res['accuracy'] == max_acc, results))
    if len(filtered_results) == 1:
        return filtered_results[0]
    # filter by minimum amount of channels used
    min_chan_amount = min(len(res['channels']) for res in filtered_results)
    filtered_results = list(filter(lambda res: len(res['channels']) == min_chan_amount, results))
    if len(filtered_results) == 1:
        return filtered_results[0]
    # random select - think about more ways
    return filtered_results[0]


def grid_search_multiple_params(params_lst: List, x_train: np.ndarray, y_train: np.ndarray) -> Dict:
    """
    Activate grid search on a list of multiple parameters options
    :param params_lst: list of dictionaries for grid search
    :param x_train:
    :param y_train:
    :return: a dictionary with the best accuracy, parameters and name
    """
    best_acc = 0
    best_params = None
    for curr_params in params_lst:
        gs = GridSearchCV(SVC(), curr_params)
        gs.fit(x_train, y_train)
        if gs.best_score_ > best_acc:
            best_acc = gs.best_score_
            best_params = gs.best_params_
        log_data('curr params search:', curr_params, 'results: ', best_acc, best_params)

    return {'accuracy': best_acc,
            'parameters': best_params,
            'name': 'SVM'}


def svm_hp_search(train_data: np.ndarray, train_labels: Union[List, np.ndarray]) -> Dict:
    """
    Hyper parameter search for svm
    :param train_data: final eeg training data with shape: #trial, #classes, sample
    :param train_labels: list of labels for each trial
    :return: dictionary with the best accuracy and parameters
    """
    x_train, y_train = final_eeg_to_train_data(train_data, train_labels)
    params_ops = [{'kernel': ['rbf', 'sigmoid'],
                   'C': [0.5, 0.75, 1, 1.25, 1.5],
                   'shrinking': [True, False]
                   },
                  {'kernel': ['poly'],
                   'C': [0.75, 1, 1.25],
                   'degree': [2, 3, 4],
                   'shrinking': [True, False]
                   },
                  ]
    return grid_search_multiple_params(params_ops, x_train, y_train)


def channel_search_general(channels_comb_lst: List, data_manipulation_func: Callable[[List], List],
                           filtered_eeg: List, search_func: Callable[[np.ndarray], Dict]):
    """
    A function that generalize the channel manipulation search
    :param channels_comb_lst: list of all channels combinations to check
    :param data_manipulation_func: function that receives a list of eeg data with
            shape: #trial, #classes, #selected_cahhnels, sample_size
            and returns a list of the eeg data after doing some manipulation on the channels axis, each eeg with
            shape:  #trial, #classes, sample_size
    :param filtered_eeg: the eeg data to apply channel selection and manipulation
    :param search_func: a function that receives the train data and train labels and applies grid search on the data,
            this function returns a dictionary with accuracy and parameter keys
    :return: A dictionary of the results of the best selected model, with keys: accuracy, parameters & channels
    """
    final_train_data = data_manipulation_func(filtered_eeg)
    model_search_results = np.array(p_map(search_func, final_train_data))
    # model_search_results = np.array(list(map(search_func, final_train_data)))
    best_accs = [res['accuracy'] for res in model_search_results]
    best_res_idx = np.argpartition(best_accs, -5)[-5:]
    final_results = model_search_results[best_res_idx]
    for idx, curr_res in enumerate(final_results):
        curr_res['channels'] = channels_comb_lst[best_res_idx[idx]]
    final_selected_model = select_best_model(final_results)
    return final_selected_model


def mean_channel_search(channels_comb: Set, processed_eeg: np.ndarray,
                        training_labels: Union[np.ndarray, List]) -> Dict:
    """
    Do channel parameter search using a mean on all channels
    :param channels_comb: set of combinations of all channels
    :param processed_eeg: the processed eeg with shape: #trial, #classes, #channels, sample
    :param training_labels: list of training labels with len: #trials
    :return: A dictionary of the results of the best selected model, with keys: accuracy, parameters, channels and
            manipulation_func - the channel manipulation function that receives a list of eeg data (after channel
            selection) and returns a list of eeg data after channel manipulation
    """
    def data_manipulation(filtered_data_):
        all_train_data = [np.mean(curr_data, axis=2) for curr_data in filtered_data_]
        return all_train_data
    log_data('mean search on channels:', channels_comb)
    filtered_data = [processed_eeg[:, :, curr_chans, :] for curr_chans in channels_comb]
    search_func = partial(svm_hp_search, train_labels=training_labels)
    search_res = channel_search_general(list(channels_comb), data_manipulation, filtered_data, search_func)
    search_res['manipulation_func'] = data_manipulation
    return search_res


def concat_channel_search(channels_comb: Set, processed_eeg: np.ndarray,
                          training_labels: Union[List, np.ndarray]) -> Dict:
    """
    Do channel parameter search by concatinating all channels
    :param channels_comb: set of combinations of all channels
    :param processed_eeg: the processed eeg with shape: #trial, #classes, #channels, sample
    :param training_labels: list of training labels with len: #trials
    :return: A dictionary of the results of the best selected model, with keys: accuracy, parameters, channels and
            manipulation_func - the channel manipulation function that receives a list of eeg data (after channel
            selection) and returns a list of eeg data after channel manipulation
    """
    def data_manipulation(filtered_data_):
        channels_first_data = [np.moveaxis(curr_eeg, 2, 0) for curr_eeg in filtered_data_]
        all_train_data = [np.concatenate(curr_data, axis=-1) for curr_data in channels_first_data]
        return all_train_data
    log_data('concat channels search on channels:', channels_comb)
    filtered_data = [processed_eeg[:, :, curr_chans, :] for curr_chans in channels_comb]
    search_func = partial(svm_hp_search, train_labels=training_labels)
    search_res = channel_search_general(list(channels_comb), data_manipulation, filtered_data, search_func)
    search_res['manipulation_func'] = data_manipulation
    return search_res


def channels_search(processed_eeg: np.ndarray, training_labels: Union[List, np.ndarray]) -> Dict:
    """
    The main function for doing hyper parameter search and channel search
    :param processed_eeg: the processed eeg with shape: #trials, #classes, #eeg channels, sample
    :param training_labels: list of training labels with len: #trials
    :return: A dictionary of the results of the best selected model, with keys: accuracy, parameters, channels and
            manipulation_func
    """
    # create channels combination search
    channels_to_use = [11, 2, 7]  # [0, 1, 2, 4, 5, 6, 9, 11, 15]
    channels_comb = [list(itertools.combinations(channels_to_use, i)) for i in range(1, len(channels_to_use))]
    channels_comb = set([y for x in channels_comb for y in x])
    # Do channels search with hp search
    channel_func_mode = [(mean_channel_search, 'mean'), (concat_channel_search, 'concat')]
    channel_search_results = list(map(lambda x: x[0](channels_comb, processed_eeg, training_labels),
                                      channel_func_mode))
    for res, (_, mode) in zip(channel_search_results, channel_func_mode):
        res['mode'] = mode
    best_model = select_best_model(channel_search_results)

    return best_model


def final_model_train(processed_eeg: np.ndarray, training_labels: Union[List, np.ndarray], results: Dict) -> Any:
    """
    A function that creates a final SVM model based on the best parameters received
    :param processed_eeg: the processed eeg with shape: #trials, #classes, #eeg channels, sample
    :param training_labels: list of training labels with len: #trials
    :param results: the search results dictionary with keys: accuracy, parameters, channels, manipulation_func
    :return: a final SVM model
    """
    if results['manipulation_func'] is not None:
        manipulated_data = results['manipulation_func']([processed_eeg[:, :, results['channels'], :]])
    else:
        manipulated_data = processed_eeg
    x_train, y_train = final_eeg_to_train_data(np.array(manipulated_data[0]), training_labels)
    model = SVC(**results['parameters'])
    model.fit(x_train, y_train)
    return model


def save_data_for_matlab(models_folder, recording_folder_path, model, result):
    """
    Save the model, search result and needed data for online in a newly created folder in models_folder
    :param models_folder: folder where a directory with the result will be saved
    :param recording_folder_path: path of folder with the recording data
    :param model: final model to save
    :param result: the final result object of the search to save
    :return: path to model save directory
    """
    model_folder_name = f'model_{result["accuracy"]:.2f}___{datetime.now().strftime("%y_%m_%d_%H_%M")}'
    save_dir = os.path.join(models_folder, model_folder_name)
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(model, os.path.join(save_dir, f'model_{result["name"]}_{result["accuracy"]:.2f}.sav'))
    with open(os.path.join(save_dir, 'search_results.json'), "w") as file:
        json.dump(result, file)
    shutil.copy(os.path.join(recording_folder_path, 'parameters.mat'), os.path.join(save_dir, 'parameters.mat'))
    return save_dir


def main_search(recording_folder_path: str, models_folder: Optional[str] = None, search_channels=True):
    """
    This function preforms hyperparameter and channel search (if selected) creates the best model and saves it to the
    folder received in folder_path
    :param recording_folder_path: path of folder with the recording data
    :param models_folder: folder for saving the model - if None model will be saved in folder_path
    :param search_channels: True if channel search is also wanted
    :return:
    """
    processed_eeg, training_labels = load_data(recording_folder_path)
    if search_channels:
        log_data('Doing channels search')
        final_result = channels_search(processed_eeg, training_labels)
    else:
        log_data('Skipping channels search')
        final_result = None

    for_log = final_result.copy()
    for_log['manipulation_func'] = 0
    log_data('Final Results:', for_log)
    final_model = final_model_train(processed_eeg, training_labels, final_result)
    if models_folder is not None:
        save_dir = save_data_for_matlab(models_folder, recording_folder_path, final_model, final_result)
    else:
        joblib.dump(final_model, os.path.join(recording_folder_path, f'model_{final_result["accuracy"]:.2f}'))
        save_dir = recording_folder_path

    print('******')
    print(save_dir)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Not enough input parameters')
        exit(-1)
    logging.basicConfig(filename=os.path.join(sys.argv[1], 'py_debug.txt'), level=logging.DEBUG)
    main_search(*sys.argv[1:])
