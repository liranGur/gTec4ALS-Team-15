import numpy as np
from numpy import mean

from config import const
from utils import load_mat_data, load_parameters
from plotly import graph_objs as go
from plotly.subplots import make_subplots


HZ = 512


def calculate_triggers_sample(triggers_time: np.ndarray, hz: int, recording_len: int) -> np.ndarray:
    end_time = triggers_time[-1]
    total_recording_time = recording_len/hz
    start_time = end_time - total_recording_time

    triggers_for_sampling = triggers_time[:-1]
    sample_idx = np.zeros(len(triggers_time))
    for idx, trigger in enumerate(triggers_for_sampling):
        time_from_start = trigger - start_time
        sample_idx[idx] = int(time_from_start * hz)
    return sample_idx


def visualize_raw_eeg_single_trial(trial_data: np.ndarray, triggers_time: np.ndarray, hz: int,
                                   triggers_classes: np.ndarray, title: str, odd_ball_class: int) -> go.Figure:
    eeg_channels = trial_data.shape[0]
    fig = make_subplots(rows=eeg_channels, cols=1)
    fig.update_layout({'title': title})
    for idx, channel in enumerate(trial_data):
        fig.add_trace(go.Scatter(y=channel, x=np.arange(0, len(channel))), row=idx+1, col=1)

    triggers_sample_idx = calculate_triggers_sample(triggers_time, hz, trial_data.shape[1])
    odd_ball_samples = triggers_sample_idx[np.argwhere(triggers_classes == odd_ball_class).ravel()]
    for sample in odd_ball_samples:
        fig.add_vline(x=sample)

    return fig


def vis_raw_data(save_folder: str):
    eeg_data = load_mat_data(const.eeg_name, save_folder)
    sequences = load_mat_data(const.training_sequences, save_folder)
    labels = load_mat_data(const.training_labels, save_folder).flatten()
    parameters = load_parameters(save_folder)
    triggers_time = load_mat_data(const.triggers_time, save_folder)

    for trial in range(eeg_data.shape[0]):
        title = f'Trial: {trial} - target class: {labels[trial]}'
        fig = visualize_raw_eeg_single_trial(eeg_data[trial], triggers_time[trial], HZ, sequences[trial], title,
                                             labels[trial])
        fig.show()
        break


def plot_raw_data_with_triggers_lines(save_folder: str, trial_to_plot: int=1, hz=512):
    def get_target_trigger_sample_idx(times, expected_class, eeg_sample_size, training_labels, hz=hz):
        end_time = times[-1]
        total_recording_time = eeg_sample_size / hz
        start_time = end_time - total_recording_time
        results = list()
        target_time = times[:-1][training_labels == expected_class]
        for curr in target_time:
            time_diff = curr - start_time
            sample_idx = int(time_diff*hz)
            results.append(sample_idx)

        return results


    eeg_data = load_mat_data(const.eeg_name, save_folder)
    sequences = load_mat_data(const.training_sequences, save_folder)
    labels = load_mat_data(const.training_labels, save_folder).flatten()
    triggers_time = load_mat_data(const.triggers_time, save_folder)
    eeg_mean = mean(eeg_data[trial_to_plot], axis=0)
    targets_sample_idx = get_target_trigger_sample_idx(triggers_time[trial_to_plot], labels[trial_to_plot],
                                                       eeg_data.shape[-1], sequences[trial_to_plot])
    plot_nums = 4
    fig = make_subplots(rows=plot_nums, cols=1)
    window_size_factor = 1
    for idx in range(plot_nums):
        sample_idx = targets_sample_idx[idx]
        start_idx = sample_idx - 50
        end_idx = int(sample_idx + hz*window_size_factor)
        fig.add_trace(go.Scatter(y=eeg_mean[start_idx:end_idx], x=np.arange(start_idx, end_idx)), row=idx+1, col=1)
        fig.add_vline(sample_idx, row=idx+1, col=1)     # trigger Line
        fig.add_vline(sample_idx+153, line_dash='dash', line_color='green', row=idx + 1, col=1)  # P300 location

    fig.update_layout({'title': 'Raw EEG with mean on channels for targets'})
    fig.show()


def visualize_split_and_mean(save_folder: str, trial_to_plot: int = 1, hz=512):
    mean_eeg = load_mat_data(('meanTriggers', 'meanTriggers'), save_folder)
    # mean_eeg = mean_eeg[trial_to_plot]
    labels = load_mat_data(const.training_labels, save_folder).flatten()
    fig = make_subplots(rows=4, cols=3)
    for trail in range(trial_to_plot, trial_to_plot+3):
        for cls in range(4):
            to_plot = mean(mean_eeg[trail, cls], axis=0)
            fig.add_trace(go.Scatter(y=to_plot, x=np.arange(len(to_plot)), name=f't:{trail}_c:{cls}'),
                          row=cls+1, col=trail+1-trial_to_plot)
            fig.add_vline(153, line_dash='dash', line_color='green', row=cls+1, col=trail+1-trial_to_plot)  # P300 location

    fig.show()
    print(labels[trial_to_plot:trial_to_plot+3])


if __name__ == '__main__':
    # plot_raw_data_with_triggers_lines('C:\\Ariel\\Files\\BCI4ALS\\gTec4ALS-Team-15\\P300\\recordingFolder\\100\\24-5_bandpass\\')
    visualize_split_and_mean('C:\\Ariel\\Files\\BCI4ALS\\gTec4ALS-Team-15\\P300\\recordingFolder\\100\\24-5_bandpass\\', 7)
