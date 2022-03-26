import numpy as np

from config import const
from utils import load_mat_data
from plotly import graph_objs as go


def vis_raw_data(save_folder: str):
    raw_data = load_mat_data(const.eeg_name, save_folder)
    sequences = load_mat_data(const.training_sequences, save_folder)
    labels = load_mat_data(const.training_labels, save_folder)
    parameters = load_mat_data(const.training_parameters, save_folder)
    time_between_triggers = parameters[const.time_pause_between_triggers]
    hz = parameters[const.hz]
    start_tick = parameters[const.start_pause_length]*hz*time_between_triggers
    ticks_between_triggers = time_between_triggers*hz

    for idx, (trail, target_class) in enumerate(zip(raw_data[0], labels)):
        odd_balls_idx = np.argwhere(sequences == target_class).ravel()
        fig = go.Figure()
        fig.update_layout({'title': f'Trail {idx+1} - Class: {target_class}'})
        fig.add_trace(go.Line(trail))
        for oddball in odd_balls_idx:
            fig.add_vline(x=start_tick + oddball*ticks_between_triggers, line_width=3, line_color="black",
                          annotation_text='odd ball trigger')
        fig.show()
