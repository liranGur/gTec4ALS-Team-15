import numpy as np

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


if __name__ == '__main__':
    vis_raw_data('C:\\Ariel\\Files\\BCI4ALS\\gTec4ALS-Team-15\\p300Recordings\\100\\26-Apr-2022 12-30-27')
