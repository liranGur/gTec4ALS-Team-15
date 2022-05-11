import os.path
from typing import Tuple, Dict
import numpy as np
import pandas as pd
import scipy.io
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from scipy.signal import decimate
from scipy.signal import butter, lfilter


HZ = 512


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bandpass(raw_eeg, lowcut=0.5, highcut=40, fs = HZ):
    flipped_raw_eeg = np.flip(raw_eeg, axis=0)
    mirror_eeg = np.concatenate((flipped_raw_eeg, raw_eeg, flipped_raw_eeg), axis=-1)
    bandpasses_eeg = np.zeros(raw_eeg.shape)
    for idx, trial in enumerate(mirror_eeg):
        filtered_mirrored = butter_bandpass_filter(trial, lowcut, highcut, fs)
        bandpasses_eeg[idx, :, :] = filtered_mirrored[:, raw_eeg.shape[-1]:raw_eeg.shape[-1] * 2]
    return bandpasses_eeg


def split_eeg_by_triggers(raw_eeg, triggers_times, pre_trigger_time, post_trigger_time):
    pass


def average_by_trigger_class(split_eeg, training_vector):
    pass


def down_sample_eeg(split_eeg, down_sample_factor):
    pass


def load_data(folder_path: str):
    names = ['EEG', 'trainingVector', 'trainingLabels', 'triggersTimes']
    raw_eeg = scipy.io.loadmat(os.path.join(folder_path, f'{names[0]}.mat'))[names[0]]
    training_vector = scipy.io.loadmat(os.path.join(folder_path, f'{names[1]}.mat'))[names[1]]
    training_labels = scipy.io.loadmat(os.path.join(folder_path, f'{names[2]}.mat'))[names[2]]
    triggers_times = scipy.io.loadmat(os.path.join(folder_path, f'{names[3]}.mat'))[names[3]]
    return raw_eeg, training_vector, training_labels, triggers_times


def preprocess(folder_path):
    low_lim = 0.5
    high_lim = 40
    pre_trigger_time = 0.2
    post_trigger_time = 1
    down_sample_factor = 60

    raw_eeg, training_vector, training_labels, triggers_times = load_data(folder_path)

    bandpass_eeg = bandpass(raw_eeg, low_lim, high_lim)
    scipy.io.savemat(os.path.join(folder_path, 'bandpass.mat'), {'bandpass': bandpass_eeg})

    split_eeg = split_eeg_by_triggers(bandpass_eeg, triggers_times, pre_trigger_time, post_trigger_time)
    scipy.io.savemat(os.path.join(folder_path, 'splitEEG.mat'), {'splitEEG': split_eeg})

    average_eeg = average_by_trigger_class(split_eeg, training_vector)
    scipy.io.savemat(os.path.join(folder_path, 'averageEEG.mat'), {'averageEEG': average_eeg})

    down_sampled_eeg = down_sample_eeg(average_eeg, down_sample_factor)
    scipy.io.savemat(os.path.join(folder_path, 'downsampledEEG.mat'), {'downsampledEEG': down_sampled_eeg})



if __name__ == '__main__':
    folder_path_ = ''
    preprocess(folder_path_)
