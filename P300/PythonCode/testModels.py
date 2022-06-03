import itertools
import os
from functools import partial
from typing import Tuple
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
from multiprocessing import Pool
from p_tqdm import p_tqdm

from utils import load_mat_data
from preprocessing import preprocess


def kfold_model(model, data, targets, splits):
    accuracy = list()
    skf = StratifiedKFold(n_splits=splits, random_state=None, shuffle=True)
    for train_idx, test_idx in skf.split(data, targets):
        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = targets[train_idx], targets[test_idx]
        model.fit(x_train, y_train)
        preds = model.predict(x_test)
        curr_acc = accuracy_score(y_test, preds)
        accuracy.append(curr_acc)

    return accuracy


def load_data(folder_path):
    processed_eeg = load_mat_data(('data_test', 'trainData'), folder_path)
    targets = load_mat_data(('data_target', 'targets'), folder_path)
    return processed_eeg, targets


def processed_eeg_to_train_data(processed_eeg, expected_classes) -> Tuple[np.ndarray, np.ndarray]:
    expected_classes = expected_classes.ravel()
    num_classes = processed_eeg.shape[1]
    num_trials = processed_eeg.shape[0]
    sample_size = processed_eeg.shape[-1]
    model_train_samples = (num_classes - 1) * num_trials
    targets = np.zeros(model_train_samples)
    train_data = np.zeros((model_train_samples, sample_size))
    for trial_idx, trial_data in enumerate(processed_eeg):
        for cls_idx, class_data in enumerate(trial_data[1:]):       # skip the idle class
            res_idx = trial_idx*(num_classes-1) + cls_idx           # The 1 is used because of skipping idle class
            train_data[res_idx] = np.mean(class_data, axis=0)
            if expected_classes[trial_idx] == cls_idx + 2:          # adding 1 because of skipping idle class and 1 because matlab indexing start with 1
                targets[res_idx] = 1

    return train_data, targets


def mean_acc_model_preprocess(model, folder_path, splits):
    data, targets = processed_eeg_to_train_data(*load_data(folder_path))
    acc = kfold_model(model, data, targets, splits)
    print(f'Accuracies are: {acc}')
    print(f'Mean accuracy: {np.mean(acc)}')


def mean_acc_model(model, folder_path, splits):
    data, targets = load_data(folder_path)
    targets = targets.ravel()
    acc = kfold_model(model, data, targets, splits)
    print(f'Accuracies are: {acc}')
    print(f'Mean accuracy: {np.mean(acc)}')


def _preprocess_helper(pre, post, ds, folder_path, expected_classes):
    best_acc = 0
    best_model_params = None
    _, _, _, down_sampled_eeg = preprocess(folder_path, pre_trigger_time=pre, post_trigger_time=post,
                                           down_sample_factor=ds)
    curr_score, curr_model_params = svm_hp_search(folder_path, down_sampled_eeg, expected_classes)
    if curr_score > best_acc:
        best_acc = curr_score
        best_model_params = curr_model_params

    return best_acc, best_model_params, {'pre': pre, 'post': post, 'ds': ds}


def search_preprocess_params(folder_path):
    expected_classes = load_mat_data(('trainingLabels', 'trainingLabels'), folder_path)
    # low_lim = [0.5]
    # high_lim = [40]
    pre_trigger_time = [-0.2, -0.1, -0.15, 0]
    post_trigger_time = [0.4, 0.5, 0.6]
    down_sample_factor = [20, 40, 60, 90]
    func = partial(_preprocess_helper, expected_classes=expected_classes, folder_path=folder_path)
    pool = Pool(os.cpu_count())
    results = list()
    for pre in pre_trigger_time:
        for post in post_trigger_time:
            for ds in down_sample_factor:
                results.append(pool.starmap(func, [[pre, post, ds]]))
    pool.close()
    pool.join()
    sorted_results = sorted(results, key=lambda res: res[0][0], reverse=True)
    best_acc, best_model_params, best_preprocess_params = sorted_results[0][0]
    return best_acc, best_model_params, best_preprocess_params


def svm_channel_search(folder_path):
    processed_eeg = load_mat_data(('processedSubtractedEEG', 'processedEEG'), folder_path)
    # processed_eeg = load_mat_data(('processedEEg', 'processedEEG'), folder_path)
    expected_classes = load_mat_data(('trainingLabels', 'trainingLabels'), folder_path)
    channels_to_use = [0, 1, 2, 4, 5, 6, 9, 11, 15]
    channels_comb = [list(itertools.combinations(channels_to_use, i)) for i in range(1, len(channels_to_use))]
    channels_comb = set([y for x in channels_comb for y in x])
    func = partial(svm_hp_search, expected_classes=expected_classes)
    all_train_data = [processed_eeg[:, :, curr_comb, :] for curr_comb in channels_comb]
    results = p_tqdm.p_map(func, all_train_data, channels_comb)
    sorted_results = sorted(results, key=lambda res: res[0], reverse=True)
    print('*'*30)
    print('*' * 30)
    print('Best results:')
    print(sorted_results[:5])
    print('*' * 30)
    print('*' * 30)

def svm_hp_search(processed_eeg, channels, expected_classes):
    data, targets = processed_eeg_to_train_data(processed_eeg, expected_classes)
    # data, targets = load_data(folder_path)
    # targets = expected_classes.ravel()
    params_ops = [{'kernel': ['rbf', 'sigmoid'],
                   'C': [0.5, 0.75, 1, 1.25, 1.5],
                   'shrinking': [True, False]
                   },
                  {'kernel': ['poly'],
                   'C': [0.75, 1, 1.25],        # , 0.75, 1, 1.1],
                   'degree': [2, 3, 4],
                   'shrinking': [True, False]
                   },
                  ]
    best_acc = 0
    best_params = None
    for params in params_ops:
        gs = GridSearchCV(SVC(), params)
        gs.fit(data, targets)
        curr_score = gs.best_score_
        if curr_score > best_acc:
            best_acc = curr_score
            best_params = gs.best_params_

    print(f'channles: {channels}, acc: {best_acc}, params: {params}')
    return best_acc, best_params, channels


if __name__ == '__main__':
    path_ = 'C:\\Ariel\\Files\\BCI4ALS\\gTec4ALS-Team-15\\P300\\recordingFolder\\100\\24-5_bandpass\\'
    # mean_acc_model(SVC(), path_, 3)
    # hp_search(path_)
    # print(search_preprocess_params(path_))
    svm_channel_search(path_)
