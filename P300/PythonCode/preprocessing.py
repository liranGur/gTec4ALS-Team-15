import os
from math import ceil
import numpy as np
import scipy.io
from scipy.signal import decimate
from scipy.signal import butter, lfilter
import warnings

HZ = 512


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bandpass(raw_eeg, lowcut=0.5, highcut=40, fs=HZ):
    flipped_raw_eeg = np.flip(raw_eeg, axis=0)
    mirror_eeg = np.concatenate((flipped_raw_eeg, raw_eeg, flipped_raw_eeg), axis=-1)
    bandpass_eeg = np.zeros(raw_eeg.shape)
    for idx, trial in enumerate(mirror_eeg):
        filtered_mirrored = butter_bandpass_filter(trial, lowcut, highcut, fs)
        bandpass_eeg[idx, :, :] = filtered_mirrored[:, raw_eeg.shape[-1]:raw_eeg.shape[-1] * 2]
    return bandpass_eeg


def split_eeg_by_triggers(raw_eeg, triggers_times, pre_trigger_time, post_trigger_time):
    num_trials, eeg_channels, trial_sample_size = raw_eeg.shape
    total_recording_time = (trial_sample_size / HZ)
    num_triggers_in_trial = triggers_times.shape[1] - 1
    pre_trigger_window_size = ceil(pre_trigger_time * HZ)
    post_trigger_window_size = ceil(post_trigger_time * HZ)
    window_size = pre_trigger_window_size + post_trigger_window_size
    trigger_move_start_time = -1 * pre_trigger_time
    bad_results = list()
    split_eeg = np.zeros((num_trials, num_triggers_in_trial, eeg_channels, window_size))

    for trial_idx, curr_trial in enumerate(raw_eeg):
        curr_trial_end_time = triggers_times[trial_idx, -1]
        first_sample_real_time = curr_trial_end_time - total_recording_time
        for trigger_idx, curr_trigger in enumerate(curr_trial):
            curr_trigger_real_time = triggers_times[trial_idx, trigger_idx]
            trigger_time_diff_from_start = curr_trigger_real_time - first_sample_real_time
            trigger_start_split_time = trigger_time_diff_from_start + trigger_move_start_time
            trigger_start_sample_idx = round(trigger_start_split_time * HZ)
            window_start_sample_idx = trigger_start_sample_idx - pre_trigger_window_size
            if window_start_sample_idx < 0:
                warnings.warn('The window start idx is less than 1 - probably recording buffer size is too small')
                bad_results.append({'trial': trial_idx, 'trigger': trigger_idx, 'start': window_start_sample_idx})
                window_start_sample_idx = 0

            window_end_sample_idx = trigger_start_sample_idx + post_trigger_window_size
            if window_end_sample_idx > trial_sample_size:
                warnings.warn('The end sample index is bigger than the EEG buffer size - probably need to add more'
                              ' time before dumping buffer')
                bad_results.append({'trial': trial_idx, 'trigger': trigger_idx, 'end': window_end_sample_idx})
                window_end_sample_idx = trial_sample_size

            curr_sample_window = raw_eeg[trial_idx, :, window_start_sample_idx:window_end_sample_idx]
            if curr_sample_window.shape[-1] < window_size:
                curr_sample_window = np.pad(curr_sample_window, ((0, 0), (0, window_size-curr_sample_window.shape[-1])))
            split_eeg[trial_idx, trigger_idx, :, :] = curr_sample_window

    return split_eeg


def average_triggers_by_class(split_eeg, training_vector):
    classes = np.unique(training_vector)
    num_trial, _, eeg_channels, sample_size = split_eeg.shape
    mean_eeg = np.zeros((num_trial, len(classes), eeg_channels, sample_size))
    for trial_idx, (eeg_trial, training_trial) in enumerate(zip(split_eeg, training_vector)):
        for curr_class in classes:
            locs = training_trial == curr_class
            mean_eeg[trial_idx, curr_class-1, :, :] = np.mean(eeg_trial[locs], axis=0)

    return mean_eeg


def down_sample_eeg(split_eeg, down_sample_factor):
    num_trial, sample_split, eeg_channels, sample_size = split_eeg.shape
    decimate_factor = HZ//down_sample_factor
    down_sample_size = ceil(sample_size/decimate_factor)
    down_sampled_eeg = np.zeros((num_trial, sample_split, eeg_channels, down_sample_size))
    for trial_idx, trial_eeg in enumerate(split_eeg):
        for split_idx, curr_split_eeg in enumerate(trial_eeg):
            down_sampled_eeg[trial_idx, split_idx, :, :] = decimate(curr_split_eeg, decimate_factor)

    return down_sampled_eeg


def load_data(folder_path: str):
    names = ['EEG', 'trainingVector', 'trainingLabels', 'triggersTimes']
    raw_eeg = list(scipy.io.loadmat(os.path.join(folder_path, f'{names[0]}.mat')).values())[-1]
    training_vector = list(scipy.io.loadmat(os.path.join(folder_path, f'{names[1]}.mat')).values())[-1]
    training_labels = list(scipy.io.loadmat(os.path.join(folder_path, f'{names[2]}.mat')).values())[-1]
    triggers_times = list(scipy.io.loadmat(os.path.join(folder_path, f'{names[3]}.mat')).values())[-1]
    return raw_eeg, training_vector, training_labels, triggers_times


def preprocess(folder_path):
    low_lim = 0.5
    high_lim = 40
    pre_trigger_time = 0.2
    post_trigger_time = 1
    down_sample_factor = 60

    raw_eeg, training_vector, training_labels, triggers_times = load_data(folder_path)

    bandpass_eeg = bandpass(raw_eeg, low_lim, high_lim)
    scipy.io.savemat(os.path.join(folder_path, 'bandpassPy.mat'), {'bandpass': bandpass_eeg})

    split_eeg = split_eeg_by_triggers(bandpass_eeg, triggers_times, pre_trigger_time, post_trigger_time)
    scipy.io.savemat(os.path.join(folder_path, 'splitEEGPy.mat'), {'splitEEG': split_eeg})

    average_eeg = average_triggers_by_class(split_eeg, training_vector)
    scipy.io.savemat(os.path.join(folder_path, 'averageEEGPy.mat'), {'averageEEG': average_eeg})

    down_sampled_eeg = down_sample_eeg(average_eeg, down_sample_factor)
    scipy.io.savemat(os.path.join(folder_path, 'downsampledEEGPy.mat'), {'downsampledEEG': down_sampled_eeg})


if __name__ == '__main__':
    folder_path_ = 'C:\\Ariel\\Files\\BCI4ALS\\gTec4ALS-Team-15\\P300\\recordingFolder\\100\\03-May-2022 12-08-47'
    preprocess(folder_path_)
