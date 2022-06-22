import itertools
import json 
import os
import sys
from functools import partial
from typing import List, Dict, Union, Callable, Set, Any, Optional
import numpy as np
import shutil
from p_tqdm import p_map
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import joblib
from datetime import datetime

from ModelSearchDataUtils import final_eeg_to_train_data, get_manipulation_func, mean_channels, concat_channels
from utils import load_mat_data, log_data, start_log
from config import Const, const


def load_data(folder_path):
    log_data(f'loading data from: {folder_path}')
    processed_eeg = load_mat_data(Const.processed_eeg, folder_path)
    training_labels = load_mat_data(Const.training_labels, folder_path)
    return processed_eeg, training_labels


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
    # "random" select - think about more ways
    return filtered_results[-1]


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
        # log_data('results: ', best_acc, best_params)

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
                   'shrinking': [True, False],
                   'probability': [True]
                   },
                  {'kernel': ['poly'],
                   'C': [0.75, 1, 1.25],
                   'degree': [2, 3, 4],
                   'shrinking': [True, False],
                   'probability': [True]
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
        log_data(f'results: {curr_res}')
    final_selected_model = select_best_model(final_results)
    return final_selected_model


def mean_channel_search(channels_comb: Set, processed_eeg: np.ndarray,
                        training_labels: Union[np.ndarray, List]) -> Dict:
    """
    Do channel parameter search using a mean on all channels
    :param channels_comb: set of combinations of all channels
    :param processed_eeg: the processed eeg with shape: #trial, #classes, #channels, sample
    :param training_labels: list of training labels with len: #trials
    :return: A dictionary of the results of the best selected model, with keys: accuracy, parameters, channels
    """
    filtered_data = [processed_eeg[:, :, curr_chans, :] for curr_chans in channels_comb]
    search_func = partial(svm_hp_search, train_labels=training_labels)
    search_res = channel_search_general(list(channels_comb), mean_channels,
                                        filtered_data, search_func)
    return search_res


def concat_channel_search(channels_comb: Set, processed_eeg: np.ndarray,
                          training_labels: Union[List, np.ndarray]) -> Dict:
    """
    Do channel parameter search by concatinating all channels
    :param channels_comb: set of combinations of all channels
    :param processed_eeg: the processed eeg with shape: #trial, #classes, #channels, sample
    :param training_labels: list of training labels with len: #trials
    :return: A dictionary of the results of the best selected model, with keys: accuracy, parameters, channels and
    """
    filtered_data = [processed_eeg[:, :, curr_chans, :] for curr_chans in channels_comb]
    search_func = partial(svm_hp_search, train_labels=training_labels)
    search_res = channel_search_general(list(channels_comb), concat_channels,
                                        filtered_data, search_func)
    return search_res


def channels_search(processed_eeg: np.ndarray, training_labels: Union[List, np.ndarray]) -> Dict:
    """
    The main function for doing hyper parameter search and channel search
    :param processed_eeg: the processed eeg with shape: #trials, #classes, #eeg channels, sample
    :param training_labels: list of training labels with len: #trials
    :return: A dictionary of the results of the best selected model, with keys: accuracy, parameters, channels
    """
    # create channels combination search
    channels_to_use = [1, 2, 4, 6, 7, 8, 11, 13, 15]  # [0, 1, 2, 4, 5, 6, 9, 11, 15]
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
    :param results: the search results dictionary with keys: accuracy, parameters, channels
    :return: a final SVM model
    """

    channel_manipulation_func = get_manipulation_func(results['mode'])
    manipulated_data = channel_manipulation_func([processed_eeg[:, :, results['channels'], :]])
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
    with open(os.path.join(save_dir, const.search_result_file), "w") as file:
        json.dump(result, file)
    shutil.copy(os.path.join(recording_folder_path, 'parameters.mat'), os.path.join(save_dir, 'parameters.mat'))
    return save_dir


def main_ensemble_search(recording_folder_path):
    processed_eeg, training_labels = load_data(recording_folder_path)
    ensemble_search_hp(processed_eeg, training_labels)


def ensemble_search_hp(processed_eeg, training_labels):
    channel_accs = list()
    best_params = list()
    selected_channels = list()
    for channel in np.arange(16):
        curr_eeg = processed_eeg[:, :, channel, :]
        curr_ans = svm_hp_search(curr_eeg, training_labels)
        if curr_ans['accuracy'] > 0.8:
            channel_accs.append(curr_ans['accuracy'])
            best_params.append(curr_ans['parameters'])
            selected_channels.append(channel)
        # channel_accs.append(curr_ans['accuracy'])
        # best_params.append(curr_ans['parameters'])

    from sklearn.model_selection import KFold
    indices = np.arange(len(training_labels))

    all_accs = list()
    for train_idx, test_idx in KFold(n_splits=3).split(indices):
        all_preds = list()
        y_true = None
        for idx, channel in enumerate(selected_channels):
            data, labels = final_eeg_to_train_data(processed_eeg[:, :, channel, :], training_labels)
            x_train = data[train_idx]
            y_train = labels[train_idx]
            x_test = data[test_idx]
            if y_true is None:
                y_true = labels[test_idx]
            assert np.all(y_true == labels[test_idx])
            curr_model = SVC(**best_params[idx])
            curr_model.fit(x_train, y_train)
            all_preds.append(curr_model.predict(x_test))

        votes = np.mean(all_preds, axis=0)
        final_preds = votes >= len(selected_channels)
        final_preds = final_preds.astype('int')
        all_accs.append(np.sum(final_preds == y_true)/len(y_true))

    print(f'{all_accs}')
    print(f'final acc: {np.mean(all_accs)}')


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

    log_data('Final Results:', final_result)
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
    start_log(True, 'train')
    main_search(*sys.argv[1:])
    # main_ensemble_search(sys.argv[1])