import os
from datetime import datetime


def get_current_time_and_date():
    return datetime.now().strftime("%d-%m_%H-%M-%H")


# ------------------ MultiProcess Options ---------------------------
DONT_USE_MULTI_PROCESS = 0
MULTI_PROCESS_PER_MODEL = 1  # each models will receive a process where it will search all folds sequentially and mean
MULTI_PROCESS_PER_FOLD = 2  # each models will open a pool for all its folds. Models are calculated Sequentially

# ------------------ Parameters Split Options ---------------------------
NO_PARAMETERS_SPLIT = -1

# ------------------ DIFFS  ----------------------------------------
USE_FILE_PRINT = False     # Set this to true only in linux Env
FAKE_PARAMS_FOR_TEST = False
SELECTED_MULTI_PROCESS_OPTION = MULTI_PROCESS_PER_MODEL

NUMBER_OF_PROCESSES = 6
FIRST_K_FOLD_SIZE = 5
GRID_SEARCH_CV = 3

RF_PARAMS_TO_USE = 0  # possible values are: 0, 1, -1(won't split the parameters)
CAT_BOOST_PARAMS_TO_USE = 0  # possible values are: 0,1
XGBOOST_PARAMS_TO_USE = 0  # possible values are: 0,1,2,3


# ------------------ Directories Configs ---------------------------
DATA_BASE = "data"
OUT_DIR = "run_results"
DATA_DIR = os.path.join(DATA_BASE, "OldRecordings-777")
SAVE_DIR = os.path.join(OUT_DIR, f"saves_{get_current_time_and_date()}")
MODELS_SAVES = "models"


# ------------------ Constants  !!!!!!! DO NOT CHANGE !!!!!  --------
EEG_TRIALS_PREPROCESSED = ("MIData.mat", "MIData")  # EEG signal during trials after pre-processing
RESTING_STATES_BANDS = ("restingStateBands.mat", "restingStateBands")  # Resting state data after pre processing
TRAIN_FEATURES = ("MIFeatures.mat", "MIFeatures")  # Trials data after feature extraction
TRAINING_LABELS = ("trainingVec.mat", "trainingVec")  # vector of the labels of each trial
TRIALS_EEG = ("EEG", "EEG")  # Raw EEG data split for each trial
TRIALS_TO_REMOVE = ("trials2remove.mat", "trials2remove")
