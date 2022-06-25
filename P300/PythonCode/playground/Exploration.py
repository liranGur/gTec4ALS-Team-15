import os
import sys
import numpy as np

from OnlineInference import load_model, get_inference_data, probabilities_decision_func


# <editor-fold desc="Probabilities exploration">
def analyze_proba(all_preds, all_probs, min_prob_th, min_diff):
    bad_prob_th = 0
    bad_prob_diff = 0
    good_probs = 0
    total_bad_preds = 0
    good_preds = 0
    probs_failed_vs_preds_idx = list()
    failed_preds_and_probs_idx = list()
    disagreements_idx = list()

    def helper(prob):
        nonlocal total_bad_preds
        nonlocal good_probs
        nonlocal bad_prob_diff
        nonlocal bad_prob_th
        if np.any(prob > min_prob_th) and abs(prob[0] - prob[1]) > min_diff:
            good_probs += 1
            return True
        else:
            if np.all(prob <= min_prob_th):
                bad_prob_th += 1
            if abs(prob[0] - prob[1]) <= min_diff:
                bad_prob_diff += 1
            return False

    for idx, (prob, preds) in enumerate(zip(all_probs, all_preds)):
        if np.sum(preds) > 1:
            total_bad_preds += 1
            if not helper(prob):
                failed_preds_and_probs_idx.append(idx)
        elif np.sum(preds) < 1:
            total_bad_preds += 1
            if not helper(prob):
                failed_preds_and_probs_idx.append(idx)
        else:
            good_preds += 1
            if np.argmax(prob) != np.argmax(preds):
                disagreements_idx.append(idx)
            if not helper(prob):
                probs_failed_vs_preds_idx.append(idx)

    return bad_prob_th, bad_prob_diff, good_probs, total_bad_preds, good_preds, \
           probs_failed_vs_preds_idx, failed_preds_and_probs_idx, disagreements_idx


def test_proba(folder_path: str, min_prob: float = 0.4, min_dif: float = 0.05):
    original_probs = list()
    all_preds = list()
    all_probs = list()
    model = load_model(folder_path)
    for file_name in os.listdir(folder_path):
        if 'rec_data' not in file_name:
            continue
        data = get_inference_data(folder_path, file_name)
        curr_probs = model.predict_proba(data)
        curr_preds = model.predict(data)
        original_probs.append(curr_probs)
        all_probs.append(curr_probs[:, 1])
        all_preds.append(curr_preds)

    bad_prob_th, bad_prob_diff, good_probs, total_bad_preds, good_preds, probs_failed_vs_preds_idx, \
            failed_preds_and_probs_idx, disagreements_idx = analyze_proba(all_preds, all_probs, min_prob, min_dif)

    bad_probs = len(all_preds) - good_probs
    print(f'{bad_probs=} , {total_bad_preds=} , {bad_prob_th=}, {bad_prob_diff=}, {good_probs=}')
    print(f'probs that failed and preds success: {probs_failed_vs_preds_idx}')
    print(f'probs that failed and preds failed: {failed_preds_and_probs_idx}')
    print(f'Disagreement indices {disagreements_idx}')
    print(f'good probs/ good preds: {good_probs/good_preds}')
    print(f'percent of good probs: {good_probs/len(all_preds)}')
    print(f'percent of good preds: {good_preds/len(all_preds)}')
    for idx, (orig, pred, prob) in enumerate(zip(original_probs, all_preds, all_probs)):
        print(f'{idx} - {pred} , {prob}, {orig.ravel()}')


def check_known_infer(folder_path: str, min_prob: float = 0.4, min_dif: float = 0.05):
    ans_dict = {-1: 'error',
                0: 'square',
                1: 'triangle'}
    for file_name in os.listdir(folder_path):
        if not file_name.startswith('rec_dat'):
            continue
        model = load_model(folder_path)
        data = get_inference_data(folder_path, file_name)
        curr_probs = model.predict_proba(data)
        ans = probabilities_decision_func(curr_probs)
        print(f'{file_name}   -   {ans_dict[ans]}')
# </editor-fold>


if __name__ == '__main__':
    check_known_infer(*sys.argv[1:])

