import os
import sys
from math import ceil
from typing import Optional, Dict
import numpy as np
from scipy.signal import resample

import utils
from config import Const
from OnlineInference import load_model, get_inference_data, probabilities_decision_func
from ModelSearch import channels_search, svm_hp_search


# <editor-fold desc="Probabilities exploration">
def analyze_proba(all_preds, all_probs, targets_diff=0.1, inner_diff=0.2, min_proba_strong=0.5, min_proba_weak=0.4):
    total_bad_preds = 0
    total_bad_probs = 0
    good_preds = 0
    good_probs = 0
    bad_prob_min_value = 0
    bad_prob_inner_diff = 0
    failed_preds_and_probs_idx = list()
    disagreements_idx = list()
    probs_failed_vs_preds_idx = list()

    def check_probs_result(probs):
        nonlocal targets_diff, inner_diff, min_proba_strong, min_proba_weak, \
            good_probs, bad_prob_min_value, bad_prob_inner_diff
        preds_result = probabilities_decision_func(probs, targets_diff=targets_diff, inner_diff=inner_diff,
                                       min_proba_strong=min_proba_strong, min_proba_weak=min_proba_weak)
        if preds_result != -1:
            good_probs += 1
        else:
            if np.all(probs[:, 1] < min_proba_weak):
                bad_prob_min_value += 1
            if np.subtract(*probs[0]) < inner_diff and np.subtract(*probs[1]) < inner_diff:
                bad_prob_inner_diff += 1
        return preds_result

    for idx, (prob, preds) in enumerate(zip(all_probs, all_preds)):
        probs_check = check_probs_result(prob)
        if np.sum(preds) > 1:
            total_bad_preds += 1
            if probs_check == -1:
                failed_preds_and_probs_idx.append(idx)
                total_bad_probs += 1
        elif np.sum(preds) < 1:
            total_bad_preds += 1
            if probs_check == -1:
                failed_preds_and_probs_idx.append(idx)
                total_bad_probs += 1
        else:
            good_preds += 1
            if probs_check != np.argmax(preds):
                disagreements_idx.append(idx)
            if probs_check == -1:
                probs_failed_vs_preds_idx.append(idx)
                total_bad_probs += 1

    return bad_prob_min_value, bad_prob_inner_diff, total_bad_preds, total_bad_probs, good_preds, good_probs, \
        probs_failed_vs_preds_idx, failed_preds_and_probs_idx, disagreements_idx


def test_proba(folder_path: str, targets_diff=0.1, inner_diff=0.2, min_proba_strong=0.5, min_proba_weak=0.4):
    original_probs = list()
    all_preds = list()
    all_probs = list()
    model = load_model(folder_path)
    rec_files = list(filter(lambda name: 'rec_data_' in name, os.listdir(folder_path)))
    for file_name in rec_files:
        data = get_inference_data(folder_path, file_name)
        curr_probs = model.predict_proba(data)
        curr_preds = model.predict(data)
        original_probs.append(curr_probs)
        all_probs.append(curr_probs)
        all_preds.append(curr_preds)

    bad_prob_min_value, bad_prob_inner_diff, total_bad_preds, total_bad_probs, good_preds, good_probs, \
        probs_failed_vs_preds_idx, failed_preds_and_probs_idx, disagreements_idx = \
        analyze_proba(all_preds, all_probs, targets_diff=targets_diff, inner_diff=inner_diff,
                      min_proba_strong=min_proba_strong, min_proba_weak=min_proba_weak)

    print(f'{total_bad_probs=} , {total_bad_preds=} , {bad_prob_min_value=}, {bad_prob_inner_diff=}, {good_probs=}')
    print(f'probs that failed and preds success: {probs_failed_vs_preds_idx}')
    print(f'probs that failed and preds failed: {failed_preds_and_probs_idx}')
    print(f'Disagreement indices {disagreements_idx}')
    print(f'good probs/ good preds: {good_probs/good_preds}')
    print(f'percent of good probs: {good_probs/len(all_preds)}')
    print(f'percent of good preds: {good_preds/len(all_preds)}')
    for idx, (orig, pred, prob) in enumerate(zip(original_probs, all_preds, all_probs)):
        print(f'{idx} ({rec_files[idx]}) - {pred} , {prob[:, 1].flatten()}, {orig.ravel()}')


def check_known_infer(folder_path: str, targets_diff=0.1, inner_diff=0.2, min_proba_strong=0.5, min_proba_weak=0.4,
                      expected_results: Optional[Dict] = None):
    import warnings
    warnings.filterwarnings("ignore")
    ans_dict = {-1: 'error',
                0: 'no',    # square / no
                1: 'yes'}   # triangle / yes
    model = load_model(folder_path)
    for file_name in os.listdir(folder_path):
        if not file_name.startswith('rec_dat'):
            continue
        data = get_inference_data(folder_path, file_name)
        curr_probs = model.predict_proba(data)
        ans = probabilities_decision_func(curr_probs, targets_diff, inner_diff, min_proba_strong, min_proba_weak)
        if expected_results is not None:
            print(f'{file_name}   -   {ans_dict[ans]} - correct pred: {(ans+1)==expected_results[file_name]}')
        else:
            print(f'{file_name}   -   {ans_dict[ans]}')
# </editor-fold>


# <editor-fold desc="search downsampling using SVM">
def explore_downsampling(data_folder: str):
    training_labels = utils.load_mat_data(Const.training_labels, data_folder)
    subtracted_mean = utils.load_mat_data(Const.subtracted_mean, data_folder)
    start, end = 10, 260
    downsampling_factor = np.linspace(start=start, stop=end, num=ceil(end/start), dtype=int)
    for curr_ds in downsampling_factor:
        print(f'downsample: {curr_ds}')
        curr_eeg = resample(subtracted_mean, curr_ds, axis=-1)
        results = channels_search(curr_eeg, training_labels, svm_hp_search)
# </editor-fold>


if __name__ == '__main__':
    check_known_infer('C:\\Ariel\\Files\\BCI4ALS\\gTec4ALS-Team-15\\P300\\recordingFolder\\500\\xgboosttesting\\model_0.90___2022_07_07_17_08')
    test_proba('C:\\Ariel\\Files\\BCI4ALS\\gTec4ALS-Team-15\\P300\\recordingFolder\\500\\xgboosttesting\\model_0.90___2022_07_07_17_08')
    # explore_downsampling('C:\\Ariel\\Files\\BCI4ALS\\gTec4ALS-Team-15\\P300\\recordingFolder\\500\\26-Jun-2022_12-42-41')
