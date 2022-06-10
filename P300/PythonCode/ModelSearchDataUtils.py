from typing import Optional, Union, List, Tuple

import numpy as np

from utils import log_data


def get_manipulation_func(mode: str):
    return {'mean': mean_channels,
            'concat': concat_channels}[mode]


def mean_channels(filtered_data):
    all_train_data = [np.mean(curr_data, axis=2) for curr_data in filtered_data]
    return all_train_data


def concat_channels(filtered_data):
    channels_first_data = [np.moveaxis(curr_eeg, 2, 0) for curr_eeg in filtered_data]
    all_train_data = [np.concatenate(curr_data, axis=-1) for curr_data in channels_first_data]
    return all_train_data


def final_eeg_to_train_data(eeg_data: np.ndarray, labels: Optional[Union[List, np.ndarray]],
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
    # TODO think about how to know if there is an idle class
    log_data(f'skipping idle class: {remove_idle_cls}')

    num_trials = eeg_data.shape[0]
    if remove_idle_cls:
        # remove idle class from data
        eeg_data = eeg_data[:, 1:, :]
    num_classes = eeg_data.shape[1]
    x_train = np.reshape(eeg_data, (num_trials * num_classes, eeg_data.shape[2]))

    if labels is not None:
        y_train = np.zeros(x_train.shape[0])
        labels = np.array(labels).ravel() - 1  # Convert to python indexing
        if remove_idle_cls:
            # This is done because the target values with idle class are from 2 to number of classes and here the
            # indices are 0 to num of classes, so we need to fix by 2 (1 for matlab index and 1 for skipping idle class)
            labels = labels - 1
        for trial_idx, curr_trial_label in enumerate(labels):
            for cls in range(num_classes):
                y_train[trial_idx*num_classes + cls] = int(curr_trial_label == cls)
    else:
        y_train = None

    return x_train, y_train
