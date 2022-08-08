import warnings
from itertools import repeat
from math import ceil
from typing import Optional, List, Union, Tuple
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils import load_mat_data, convert_triggers_times_to_sample_idx, convert_sample_idx_to_trigger_time
from config import const


DEF_ROWS = 4
DEF_COLS = 4
DEF_HZ = 512
DEF_CHANNELS = np.arange(16)


def _get_triggers_vline(triggers_samples_idx, training_labels):
    colors = ['black', 'blue', 'red']
    ans = [(idx, colors[cls-1], str(cls)) for idx, cls in zip(triggers_samples_idx, training_labels)]
    return ans


def _get_plot_x_vals(curr_data, convert_to_times: bool, trigger_sample_idx: int, hz: int = DEF_HZ):
    if convert_to_times:
        x_vals = convert_sample_idx_to_trigger_time(curr_data, trigger_sample_idx, hz)
    else:
        if trigger_sample_idx == 0 or trigger_sample_idx == -1:
            x_vals = np.arange(len(curr_data))
        else:
            post_x_vals = np.arange(len(curr_data) - trigger_sample_idx)
            pre_x_vals = np.flip(np.arange(-1, -(len(curr_data) - len(post_x_vals) + 1), step=-1))
            x_vals = np.concatenate((pre_x_vals, post_x_vals))
    return x_vals


def my_subplots(data: np.ndarray, convert_to_times: bool, subplot_titles: Optional[List[str]], title: str,
                trigger_sample_idx: int, v_lines: Optional[List[Tuple[Union[int, str]]]], legends_names: Optional[List[str]],
                rows: int, cols: int, hz: int = DEF_HZ):
    x_title = 'sample index' if not convert_to_times else 'time since trigger'
    v_lines = list() if v_lines is None else v_lines
    if legends_names is None:
        legends_names = [None] * (rows*cols)
    elif len(legends_names) < rows*cols:
        legends_names += [None] * (len(legends_names) - rows*cols)

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=subplot_titles, x_title=x_title)
    fig.update_layout({'title': title})

    # if rows*cols < data.shape[0]:
    #     data = np.expand_dims(data, axis=0)
    for row_idx in range(1, rows+1):
        for col_idx in range(1, cols+1):
            data_idx = (row_idx-1)*cols + col_idx - 1
            if data_idx >= data.shape[0]:
                break
            curr_data = data[data_idx]
            if len(curr_data.shape) != 1:   # in case the data has more than 1 channel for the same subplot
                for inner_idx, inner_data in enumerate(curr_data):
                    x_vals = _get_plot_x_vals(inner_data, convert_to_times, trigger_sample_idx, hz)
                    fig.add_trace(go.Scatter(x=x_vals, y=inner_data,
                                             name=legends_names[data_idx*curr_data.shape[0] + inner_idx]),
                                  row=row_idx, col=col_idx,)
            else:
                x_vals = _get_plot_x_vals(curr_data, convert_to_times, trigger_sample_idx, hz)
                fig.add_trace(go.Scatter(x=x_vals, y=curr_data, name=legends_names[data_idx]),
                              row=row_idx, col=col_idx)

            for curr_line in v_lines[:5]:
                line_x = curr_line[0]
                line_kwargs = {'x': line_x / hz if convert_to_times else line_x, 'line_color': curr_line[1]}
                if len(curr_line) == 3:
                    line_kwargs['annotation_text'] = curr_line[2]
                fig.add_vline(**line_kwargs, line_width=1, row=row_idx, col=col_idx)

    fig.show()


def _trials_param_set_up(data: np.ndarray, trials: Optional[Union[int, List[int]]]):
    trials = np.arange(data.shape[0]) if trials is None else trials
    trials = [trials] if isinstance(trials, int) else trials
    return trials


def plot_raw_eeg(folder_path: str, sample_range: Optional[List[int]], time_range: Optional[List[float]],
                 trials: Optional[Union[int, List[int]]], split_trials_to_multiple_plots: bool, convert_to_times: bool,
                 split_channels: bool, draw_triggers: bool, skip_base_class_lines: bool = False,
                 triggers_classes_to_draw: Optional[List[int]] = None,
                 channels_to_use: Union[List, np.ndarray] = DEF_CHANNELS,
                 rows: int = DEF_ROWS, cols: int = DEF_COLS, hz: int = DEF_HZ):
    """
    Plot raw EEG. Allows plotting each trial in separate plot and then each subplot can be each channel or all channels
    can be drawn on the same subplot. If multiple trials are shown in the same plot it can plot in each subplot the same
    channel of all the trials of plot each trial into different subplot.
    :param folder_path: matlab recording data folder
    :param sample_range: plot only part of the eeg between samples indices - list of 2 ints
    :param time_range:  plot only part of the eeg between time indices - list of 2 floats
    :param trials: trials to plot if none will plot all trials
    :param split_trials_to_multiple_plots: will it plot multiple plots one for each trial
    :param convert_to_times: will the x-axis will be time or sample index
    :param split_channels: plot all channels of same trial in single plot or use supplots for each channel
    :param draw_triggers: should plot line for triggers
    :param skip_base_class_lines: plot lines triggers that aren't base class
    :param triggers_classes_to_draw: plot specific triggers class
    :param channels_to_use: which channels to plot
    :param rows:
    :param cols:
    :param hz:
    :return:
    """
    if sample_range is not None and time_range is not None:
        raise ValueError("Time range and sample range can't both have values")

    raw_eeg_data = load_mat_data(const.eeg_name, folder_path)
    trials = _trials_param_set_up(raw_eeg_data, trials)
    if sample_range is not None:
        start_idx, end_idx = sample_range
    elif time_range is not None:
        start_idx, end_idx = (int(time_range[0]*hz), int(time_range[1]*hz))
    else:
        start_idx, end_idx = (0, raw_eeg_data.shape[-1])
    raw_eeg_data = raw_eeg_data[..., start_idx: end_idx]

    if draw_triggers:
        triggers_times = load_mat_data(const.triggers_time, folder_path)
        triggers_samples = convert_triggers_times_to_sample_idx(triggers_times, hz)
        triggers_samples = triggers_samples[start_idx: end_idx]
        training_labels = load_mat_data(const.training_vector, folder_path)
        triggers_idx = training_labels != 1 if skip_base_class_lines else np.ones(triggers_times.shape, dtype=bool)[:, 1:]
        if triggers_classes_to_draw is not None:
            for cls in triggers_classes_to_draw:
                triggers_idx = np.logical_and(triggers_idx, training_labels == cls)

        relevant_triggers = triggers_samples[triggers_idx]
        relevant_classes = training_labels[triggers_idx]
        v_lines = _get_triggers_vline(relevant_triggers.ravel(), relevant_classes.ravel())
    else:
        v_lines = None

    if split_trials_to_multiple_plots:
        for curr_trial in trials:
            if not split_channels:
                rows = 1
                cols = 1
                subplot_titles = []
            else:
                subplot_titles = [f'channel: {chan}' for chan in channels_to_use]
                if len(channels_to_use) < rows * (cols - 1):
                    cols = ceil(len(channels_to_use) / rows)
                if len(channels_to_use) < (rows - 1) * cols:
                    rows = ceil(len(channels_to_use) / cols)
            my_subplots(raw_eeg_data[curr_trial, channels_to_use], convert_to_times=convert_to_times,
                        trigger_sample_idx=0, subplot_titles=subplot_titles, legends_names=None,
                        title=f'Raw EEG trial: {curr_trial}', v_lines=v_lines, rows=rows, cols=cols, hz=hz)
    else:
        if not split_channels:
            rows = len(trials)
            cols = 1
        my_subplots(raw_eeg_data[trials][:, channels_to_use], convert_to_times=convert_to_times,
                    trigger_sample_idx=0, subplot_titles=[f'trial: {trial}' for trial in trials],
                    title=f'Raw EEG for trials {",".join([str(x) for x in trials])} on channels:'
                          f' {", ".join([str(x) for x in channels_to_use])}', v_lines=v_lines,
                    legends_names=None, rows=rows, cols=cols, hz=hz)


def _set_trigger_sample_for_split_data(trigger_sample_idx: Optional[int], trigger_time_from_split_start: Optional[float],
                                       hz: int = DEF_HZ):
    if trigger_sample_idx is not None and trigger_time_from_split_start is not None:
        warnings.warn('Both trigger_sample_idx & trigger_time_from_split_start - Will use trigger_sample idx for '
                      'marking the trigger')

    if trigger_sample_idx is None:
        if trigger_time_from_split_start is not None:
            trigger_sample_idx = int(ceil(trigger_time_from_split_start * hz))
        else:
            trigger_sample_idx = -1
    return trigger_sample_idx


def _set_triggers_indices_for_split_data(data: np.ndarray, training_labels: np.ndarray,
                                         triggers_indices: Optional[Union[List[int], int]], triggers_classes: Optional[Union[List[int], int]], ):
    if triggers_indices is None:
        if triggers_classes is None:
            triggers_indices = np.arange(data.shape[1])
        else:
            triggers_classes = [triggers_classes] if isinstance(triggers_classes, int) else triggers_classes
            triggers_indices = np.argwhere(np.isin(training_labels, triggers_classes)).ravel()
    elif isinstance(triggers_indices, int):
        triggers_indices = [triggers_indices]
    return triggers_indices


def _set_v_lines_for_split_data(draw_trigger_line: bool, training_labels: np.ndarray, trigger_sample_idx: int,
                                triggers_indices: np.ndarray):
    # Because when trigger index exists it will become sample with index 0 in plots the sets the x value of line to 0
    if draw_trigger_line and trigger_sample_idx <= 0:
        warnings.warn("Can't draw a trigger line when the trigger sample index is unknown (0 or 1)")
        v_lines = None
    else:
        v_lines = _get_triggers_vline(repeat(0), training_labels[triggers_indices].ravel()) \
            if draw_trigger_line else None
    return v_lines


def plot_raw_split(folder_path: str, trial: int, convert_to_times: bool, trigger_sample_idx: Optional[int],
                   trigger_time_from_split_start: Optional[float], draw_trigger_line: bool,
                   triggers_indices: Optional[Union[List[int], int]], triggers_classes: Optional[Union[List[int], int]],
                   plot_for_trigger: bool = False, subplot_same_channel: bool = False,
                   channels_to_use: Union[List, np.ndarray] = DEF_CHANNELS,
                   rows: int = DEF_ROWS, cols: int = DEF_COLS, hz: int = DEF_HZ):
    """
    plot raw EEG data split by trial from matlab data (i.e., loads split data).
    It can plot all triggers in same plot or split each trigger for each plot. If all triggers are plotted in the same
    plot it has 2 options:
        1) each subplot will contain the same channel of all the triggers, i.e., subplot 1,1 will have the first channel
         of all triggers plotted
        2) each subplot will contain all requested channels of the same trigger and each subplot will belong to a
         different trigger, i.e., subplot 1 will contain 16 channels of trigger 1 and subplot 2 will contain 16 channels
         of trigger 2 and so on
    :param folder_path: matlab recording data folder
    :param trial: selected trial for plots
    :param convert_to_times: will the x-axis will be time or sample index
    :param trigger_sample_idx: idx of the trigger in the split window
    :param trigger_time_from_split_start: time from start of split to trigger
    :param draw_trigger_line: draw v-lines on subplot for each trigger
    :param triggers_indices: triggers to plot
    :param triggers_classes: plot triggers for selected classes
    :param plot_for_trigger: for each trigger create its own plot, False will create a single plot for all triggers
    :param subplot_same_channel: will plot all channels in the same subplot (option 1 from above), False option 2
    :param channels_to_use: selected channels to plot
    :param rows:
    :param cols:
    :param hz:
    :return:
    """
    split_eeg_data = load_mat_data(const.split_raw_eeg, folder_path)
    training_labels = load_mat_data(const.training_vector, folder_path)[trial]
    trigger_sample_idx = _set_trigger_sample_for_split_data(trigger_sample_idx, trigger_time_from_split_start, hz)
    triggers_indices = _set_triggers_indices_for_split_data(split_eeg_data, training_labels, triggers_indices, triggers_classes)
    v_lines = _set_v_lines_for_split_data(draw_trigger_line, training_labels, trigger_sample_idx, triggers_indices)
    '
    if plot_for_trigger:
        for idx, curr_trigger in enumerate(triggers_indices):
            title = f'Split EEG trial: {trial} and trigger: {curr_trigger}'
            subplot_titles = [f'Channel: {chan}' for chan in channels_to_use]
            curr_v_line = None if v_lines is None else [v_lines[idx]]
            my_subplots(split_eeg_data[trial][curr_trigger, channels_to_use], convert_to_times=convert_to_times,
                        subplot_titles=subplot_titles, legends_names=None,
                        title=title, trigger_sample_idx=trigger_sample_idx, v_lines=curr_v_line, rows=rows, cols=cols)
    elif subplot_same_channel:
        title = f'Split EEG - trial: {trial} and triggers: {", ".join([str(x) for x in triggers_indices])}'
        subplot_titles = [f'Channel: {chan}' for chan in channels_to_use]
        data_for_plot = split_eeg_data[trial][triggers_indices].squeeze()[..., channels_to_use, :]
        if len(data_for_plot.shape) == 3:
            data_for_plot = np.moveaxis(data_for_plot, 0, 1)
        else:
            data_for_plot = np.expand_dims(data_for_plot, axis=0)
        # because each subplot can contain data from multiple triggers, remove class affiliation from v_line
        curr_v_line = None if v_lines is None else [(0, 'orange')]
        legends_names = [f'trigger: {trig}, channel: {chan}' for chan in channels_to_use for trig in triggers_indices]
        my_subplots(data_for_plot, convert_to_times=convert_to_times, subplot_titles=subplot_titles, title=title,
                    legends_names=legends_names,
                    trigger_sample_idx=trigger_sample_idx, v_lines=curr_v_line, rows=rows, cols=cols, hz=hz)
    else:
        title = f'Split EEG - trial: {trial} and triggers: {", ".join([str(x) for x in triggers_indices])}'
        subplot_titles = [f'Trigger: {trigger}' for trigger in triggers_indices]
        data_to_plot = split_eeg_data[trial][triggers_indices].squeeze()[..., channels_to_use, :].squeeze()
        if len(data_to_plot.shape) == 1:
            data_to_plot = np.array([data_to_plot])
            legends_names = None  # single channel no legend needed
        else:
            legends_names = [f'trigger: {trig}, channel: {chan}' for trig in triggers_indices for chan in channels_to_use]

        my_subplots(data_to_plot, convert_to_times=convert_to_times,
                    subplot_titles=subplot_titles, title=title, trigger_sample_idx=trigger_sample_idx, v_lines=v_lines,
                    rows=rows, cols=cols, legends_names=legends_names)


def plot_mean_triggers(folder_path: str, trials: Optional[Union[int, List[int]]], convert_to_times: bool, plot_for_each_trial: bool, subplot_same_channel: bool,
                       trigger_sample_idx: Optional[int], trigger_time_from_split_start: Optional[float], draw_trigger_line: bool,
                       triggers_indices: Optional[Union[List[int], int]], triggers_classes: Optional[Union[List[int], int]],
                       channels_to_use: Union[List, np.ndarray] = DEF_CHANNELS,
                       rows: int = DEF_ROWS, cols: int = DEF_COLS, hz: int = DEF_HZ):
    # split all channels for single trigger
    # plot muListiple channels on same for all classes of the same trial
    # plot single channel for all classes of same trial
    # plot muListiple channels on same for same class on muListiple trials
    # plot single channel for same class on muListiple trials
    mean_eeg_data = load_mat_data(const.mean_split_eeg, folder_path)
    trials = _trials_param_set_up(mean_eeg_data, trials)
    training_labels = np.array([np.arange(mean_eeg_data.shape[1]) for _ in range(len(trials))])
    trigger_sample_idx = _set_trigger_sample_for_split_data(trigger_sample_idx, trigger_time_from_split_start)
    triggers_indices = _set_triggers_indices_for_split_data(mean_eeg_data, training_labels, triggers_indices, triggers_classes)
    v_lines = _set_v_lines_for_split_data(draw_trigger_line, training_labels, trigger_sample_idx, triggers_indices)

    if plot_for_each_trial:
        pass
    elif subplot_same_channel:
        pass
    else:
        pass




def plot_processed_eeg():
    # the same as plot mean triggers options
    pass


def test():
    folder_path = 'C:\\Ariel\\Files\\BCI4ALS\\gTec4ALS-Team-15\\P300\\recordingFolder\\500\\26-Jun-2022_13-08-42-yes-no'
    # plot_raw_eeg(folder_path, sample_range=None, time_range=[0, 2], trials=[0, 1], convert_to_times=False,
    #              split_trials_to_multiple_plots=False, split_channels=False, draw_triggers=True, skip_base_class_lines=True)
    plot_raw_split(folder_path, trial=1, convert_to_times=True, subplot_same_channel=False, plot_for_trigger=True,
                   trigger_sample_idx=150, trigger_time_from_split_start=None, draw_trigger_line=True,
                   triggers_indices=[20, 33], triggers_classes=None)

if __name__ == '__main__':
    test()
