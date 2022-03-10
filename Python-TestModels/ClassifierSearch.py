import functools
import os.path
import sys
import warnings
from collections import OrderedDict
from multiprocessing import Pool
import json
from typing import Dict, List, Type, Union, Callable, Iterable
import _pickle as cpkl

import numpy as np
from sklearn.model_selection import StratifiedKFold

import utils
from config import *
from utils import load_data, remove_trials, get_print_func
from Classifiers import BaseClassifierSearch, RandomForestSearch, SVCSearch, SGDSearch, LDASearch, XgBoostSearch, \
    CatBoostSearch

warnings.filterwarnings("ignore")

# ==========================================================================================
# data explanation:
#   MIFeatures.mat - features extracted from
#   FeatureParam.mat - number array 4 : 0.1 : 40
#   trainingVec.mat - list of the classes during training
#   restingStateBands.mat - power of each electrode during the resting periods
#   parameters.mat - different parameters used during training
# ==========================================================================================

ALL_POSSIBLE_CLASSIFIERS = [RandomForestSearch, SVCSearch, SGDSearch, LDASearch, XgBoostSearch, CatBoostSearch]
print = get_print_func(str(sys.argv[1:]))


def load_train_data(data_folder: str):
    data = load_data(TRAIN_FEATURES, folder=data_folder)
    labels = load_data(TRAINING_LABELS, folder=data_folder)
    labels = labels.reshape(-1, )
    try:
        trials_to_remove = load_data(TRIALS_TO_REMOVE, folder=data_folder)
        trials_to_remove = trials_to_remove.reshape(-1, )
    except FileExistsError:
        trials_to_remove = list()
    return data, labels, trials_to_remove


def crete_folds(num_of_folds: int, data: np.ndarray, labels: np.ndarray) -> List[Dict]:
    """
    :param num_of_folds:
    :param data:
    :param labels:
    :return: list each element is a dictionary with the data of another fold
    """
    skf = StratifiedKFold(n_splits=num_of_folds)
    split_data = list()
    for train_index, test_index in skf.split(data, labels):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        split_data.append({
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test
        })

    return split_data


def init_saves(save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    configs = utils.get_all_configs()
    to_print = [f"{k}: {v}" for k, v in configs.items()]
    print(to_print)
    to_print = "\n".join(to_print)
    with open(os.path.join(save_dir, "configs.txt"), "w") as f:
        f.write(to_print)


def save_model_data(save_dir: str, save_info: str, model_name: str, accuracy: float,
                    parameters: Union[Dict, List[Dict]], model: BaseClassifierSearch, extra_json_data=None):
    str_acc = f"{accuracy*100:.2f}"
    str_acc = str_acc.replace(".", "_")
    save_path = os.path.join(save_dir, model_name, f"{save_info}-acc-{str_acc}")
    os.makedirs(save_path, exist_ok=True)
    json_file_path = os.path.join(save_path, "parameters.json")
    with open(json_file_path, "w") as f:
        json.dump({"accuracy": accuracy}, f)
        f.write("\n")
        json.dump(parameters, f)
        if extra_json_data is not None:
            f.write("\n")
            json.dump(extra_json_data, f)
            f.write("\n")

    if model is not None:
        model_file = os.path.join(save_path, f"{model_name}.pkl")
        with open(model_file, "wb") as f:
            cpkl.dump(model, f)


def find_best_params_single_model_fold(fold_data: Dict, model: Type[BaseClassifierSearch]):
    model = model()
    model.search_parameters(fold_data["x_train"], fold_data["y_train"])
    acc = model.model_test_acc(fold_data["x_test"], fold_data["y_test"])
    return model.name, acc, model.get_best_params(), model


def keep_unique_hps(all_hps: List[Dict]) -> List[Dict]:
    hp_no_dups = list()
    for curr_hp in all_hps:
        if curr_hp not in hp_no_dups:
            hp_no_dups.append(curr_hp)

    return hp_no_dups


def calc_mean_acc_for_model(model: Type[BaseClassifierSearch], folded_data: List[Dict], all_hps: List[Dict]):
    """
    Calculates the mean acc over all the data for all possible hyper-parameters
    :param model:
    :param folded_data:
    :param all_hps:
    :return: best mean accuracy, best hyper-parameters, all mean accuracies, all hyper-parameters tested by order
    """
    unique_hps = keep_unique_hps(all_hps)
    all_accs = list()
    best_hp = list()
    best_acc = 0
    for curr_hp in unique_hps:
        total_acc = 0
        for curr_fold in folded_data:
            curr_model = model()
            curr_model.create_model_and_train(curr_hp, **curr_fold)
            acc = curr_model.model_test_acc(**curr_fold)
            total_acc += acc

        mean_acc = total_acc/len(folded_data)
        all_accs.append(mean_acc)
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_hp.append(curr_hp)
        elif mean_acc == best_acc:
            best_hp.append(curr_hp)

    return best_acc, best_hp, all_accs, unique_hps


def find_best_params_for_model(model: Type[BaseClassifierSearch], folded_data: List[Dict],
                               save_dir: str):
    """
    Finds the best hyper-parameters for a specific model on all folds
    It saves all the models and their info from the grid search in each fold and saves the final results as well
    :param model:
    :param folded_data:
    :param save_dir:
    :return:
    """
    print(f"Finding best hyper parameters for {model.name}")
    all_hps = list()
    folds_results = run_as_multi_process(MULTI_PROCESS_PER_FOLD, find_best_params_single_model_fold, folded_data,
                                         {"model": model})
    for fold_num, (name, curr_acc, curr_hp, curr_model) in enumerate(folds_results):
        print(f"In fold {fold_num} for model: {name}")
        save_model_data(save_dir, f"fold_num_{fold_num}", name, curr_acc, curr_hp, curr_model)
        all_hps.append(curr_hp)

    # for fold_num, fold_data in enumerate(folded_data):
    #     name, curr_acc, curr_hp, curr_model = (find_best_params_single_model_fold(model, **fold_data))
    #     save_model_data(save_dir, f"fold_num_{fold_num}", name, curr_acc, curr_hp, curr_model)
    #     all_hps.append(curr_hp)

    best_mean_acc, best_hp, all_mean_accs, all_hp = calc_mean_acc_for_model(model, folded_data, all_hps)
    save_model_data(save_dir, "mean", model.name, best_mean_acc, best_hp, None,
                    dict(all_mean_accs=all_mean_accs, all_hps=all_hp))


def run_as_multi_process(mp_option: int, func: Callable, iterable: Iterable, other_args: Dict=dict(),
                         extra_print: str = "") -> List:
    results = list()
    if SELECTED_MULTI_PROCESS_OPTION != mp_option:
        for curr in iterable:
            results.append(func(curr, **other_args))
    else:
        if extra_print != "":
            print(extra_print)
        pool = Pool(NUMBER_OF_PROCESSES)
        results = pool.map_async(functools.partial(func, **other_args), iterable)
        pool.close()
        results.wait()
        results = results.get()

    return results


def main(models_to_keep: List = tuple(), minimum_improvement_required: float = 0.05,
         data_folder: str = DATA_DIR, save_dir: str = SAVE_DIR,
         base_model_name: str = "LinearDiscriminantAnalysis"):
    data, labels, trials_to_remove = load_train_data(data_folder)
    data, labels = remove_trials(data, labels, trials_to_remove)
    folded_data = crete_folds(FIRST_K_FOLD_SIZE, data, labels)
    if models_to_keep is not None and len(models_to_keep) != 0:
        models_to_search = list(filter(lambda model: model.name in models_to_keep, ALL_POSSIBLE_CLASSIFIERS))
    else:
        models_to_search = ALL_POSSIBLE_CLASSIFIERS

    init_saves(save_dir)
    run_as_multi_process(MULTI_PROCESS_PER_MODEL, find_best_params_for_model, models_to_search,
                         {"folded_data": folded_data, "save_dir": save_dir},
                         "Running mutli-process for each fold")
    print("*"*20)
    print("Finished")


if __name__ == '__main__':
    print(f"Models Received: {sys.argv}")
    main(sys.argv[1:])


# def grid_search(parameters: List[Dict], model, data: np.ndarray, labels: np.ndarray):
#     grid_clf = GridSearchCV(model, parameters, scoring="accuracy", cv=3)
#     grid_clf.fit(data, labels)
#     return grid_clf
#
#
# def train_rf(data: np.ndarray, labels: np.ndarray):
#     def create_rf_dict_params(n_estimators, max_depth, criterion, n_jobs=4):
#         return {k: [v] if type(v) is not list else v for k, v in locals().items()}
#
#     parameters = [create_rf_dict_params([i for i in range(500, 2000, 100)],
#                                         [i for i in range(5, 40, 5)] + [i for i in range(50, 600, 50)],
#                                         ["gini", "entropy"])
#                   ]
#
#     return grid_search(parameters, RandomForestClassifier(), data, labels)
#
#
# def train_svc(data: np.ndarray, labels: np.ndarray):
#     def create_svc_parameters(C, kernel, degree, shrinking, gamma, coef0):
#         return {k: [v] if type(v) is not list else v for k, v in locals().items()}
#     parameters = [create_svc_parameters([0, 0.5, 0.9, 1, 1.1, 1.5],
#                                         ["linear", "poly", "rbf", "sigmoid"],
#                                         [2, 3, 4, 5, 6],
#                                         [True, False],
#                                         ["scale", "auto"],
#                                         [0.0, 0.1, 0.01, 0.001, 0.2, 0.02, 0.3, 0.4])]
#     return grid_search(parameters, SVC(), data, labels)
#
#
# def train_sgd(data: np.ndarray, labels: np.ndarray):
#     def create_sgd_parameters(loss, penalty, alpha, n_jobs=4):
#         return {k: [v] if type(v) is not list else v for k, v in locals().items()}
#     parameters = [create_sgd_parameters(["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
#                                         ["l2", "l1", "elasticnet"],
#                                         [1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5])
#                   ]
#     return grid_search(parameters, SGDClassifier(), data, labels)
#
#
# def train_lda(data: np.ndarray, labels: np.ndarray):
#     def create_lda_parameters(solver, shrinkage):
#         return {k: [v] if type(v) is not list else v for k, v in locals().items()}
#     parameters = [create_lda_parameters(["svd", "lsqr", "eigen"],
#                                         [None, "auto", 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 1])
#                   ]
#     return grid_search(parameters, LinearDiscriminantAnalysis(), data, labels)
#
#
# def train_xgboost(data: np.ndarray, labels: np.ndarray):
#     def create_xgb_parameters(n_estimators, max_depth, learning_rate, booster, tree_method, gamma, subsample,
#                               verbosity=0, n_jobs=4):
#         return {k: [v] if type(v) is not list else v for k, v in locals().items()}
#     parameters = [create_xgb_parameters([i for i in range(100, 2000, 100)],   # TODO decrease number of parameters  / Send XGB to gpu
#                                         [None] + [i for i in range(5, 30, 5)] + [i for i in range(30, 600, 50)],
#                                         [0.1 ** i for i in range(1, 6, 1)] + [0.5 ** i for i in range(1, 4, 1)],
#                                         ["gbtree", "gblinear", "dart", "gpu_hist"],
#                                         ["exact", "approx", "hist"],
#                                         [1-0.5*i for i in range(0, 10)],
#                                         [0.1 * i for i in range(0, 11)] + [1.5, 2, 2.5, 3])
#                   ]
#
#     return grid_search(parameters, XGBClassifier(), data, labels)
#
#
# def train_catboost(data: np.ndarray, labels: np.ndarray):
#     @dataclass
#     class CatboostAdapter:
#         """
#         Adapter for catboost to comply with GridSearchCV needed interface
#         """
#         model: CatBoostClassifier
#         best_params_: Dict
#
#         def __init__(self, model, params):
#             self.model = model
#             self.best_params_ = params
#
#         def predict(self, x):
#             return self.model.predict(x)
#
#     def create_catboost_parameters(iterations, learning_rate, depth, l2_leaf_reg, border_count,
#                                    logging_level="Silent"):
#         return {k: [v] if type(v) is not list else v for k, v in locals().items()}
#
#     parameters = [create_catboost_parameters([i for i in range(50, 501, 10)],
#                                              [0.1 ** i for i in range(1, 6, 1)] + [0.5 ** i for i in range(1, 4, 1)],
#                                              [i for i in range(15)],
#                                              [i for i in range(1, 256)],
#                                              [1 - 0.1*i for i in range(0, 10)])
#                   ]
#     search_ans = CatBoostClassifier().grid_search(parameters, X=data, y=labels, cv=3, verbose=False)
#     model = CatBoostClassifier(**search_ans['params'])
#     model.fit(data, labels)
#     return CatboostAdapter(model, search_ans['params'])
#
#
# def get_best_model_for_fold(x_train, y_train, x_test, y_test, base_model_name: str,
#                             base_model_train_func: Callable[[np.ndarray, np.ndarray], Any]):
#     """
#     Finds the best model for some data using grid search
#     :param x_train:
#     :param y_train:
#     :param x_test:
#     :param y_test:
#     :param base_model_name: name of the base model for comparison
#     :param base_model_train_func:
#     :return:
#     """
#     base_model = base_model_train_func(x_train, y_train)
#     base_report = classification_report(y_test, base_model.predict(x_test), output_dict=True)
#     base_acc = base_report[ACC]
#     # Keys in this dictionary must be the same as the call to the model, i.e., globals()[<name>] should work
#     curr_bests = {"RandomForestClassifier": train_rf(x_train, y_train),
#                   "SVC": train_svc(x_train, y_train),
#                   "SGDClassifier": train_sgd(x_train, y_train),
#                   "XGBClassifier": train_xgboost(x_train, y_train),
#                   "CatBoostClassifier": train_catboost(x_train, y_train)
#                   }
#     best_acc, best_name, best_model = base_acc, base_model_name, base_model
#     for model_name, model in tqdm(curr_bests.items()):
#         report = classification_report(y_test, model.predict(x_test), output_dict=True)
#         acc = report[ACC]
#         if acc > best_acc:
#             best_acc = acc
#             best_name = model_name
#             best_model = model
#     return best_acc, best_name, best_model.best_params_, base_acc, base_model.best_params_
#
#
# def eval_model_on_all_folds(split_data, model_parameters, model_name):
#     """
#     Calculate mean accuracy of a model with specific parameters on all the different folds.
#     For each fold a model is trained and test accuracy is calculated to get final mean accuracy
#     :param split_data:
#     :param model_parameters:
#     :param model_name:
#     :return: mean accuracy, list of all models created, list of all models accuracies
#     """
#     acc = 0
#     total_folds = len(split_data)
#     models = list()
#     all_acc = list()
#     for fold in split_data:
#         x_train, x_test, y_train, y_test = itemgetter("x_train", "x_test", "y_train", "y_test")(fold)
#         model = globals()[model_name](**model_parameters)
#         model.fit(x_train, y_train)
#         y_preds = model.predict(x_test)
#         curr_acc = accuracy_score(y_test, y_preds)
#         models.append(model)
#         all_acc.append(curr_acc)
#         acc += curr_acc
#
#     return acc/total_folds, models, all_acc
#
#
# def save_all_data(save_dir_path: str, model_name: str, model_parameters: Dict, models: List, accuracies: List,
#                   mean_acc: float, extra: Any = None):
#     """
#     Save all data from training,
#     Creates a directory for current data and dumps all data to that location
#     :param save_dir_path
#     :param model_name:
#     :param model_parameters:
#     :param models: List of all the models created for mean accuracy calculation, last model is a model trained on
#     all data
#     :param accuracies: List of accuracies of the models created for mean accuracy calculation
#     :param mean_acc:
#     :param extra: any other thing to print to data file
#     :return:
#     """
#     curr_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
#     save_dir = os.path.join(save_dir_path, f"{model_name}_{mean_acc:.3f}--{curr_time}")
#     try:
#         os.mkdir(save_dir)
#     except FileExistsError:
#         pass
#     # save models
#     for idx, (model, acc) in enumerate(zip(models[:-1], accuracies)):
#         model_path = os.path.join(save_dir, f"model_{acc}_{idx}.pkl")
#         cpkl.dump(model, open(model_path, "wb"))
#     cpkl.dump(models[-1], open(os.path.join(save_dir, "full_train.pkl"), "wb"))
#     # save meta data
#     with open(os.path.join(save_dir, f"data_{mean_acc:.3f}_{curr_time}.json"), "w") as file:
#         json.dump(model_parameters, file)
#         file.write("\n")
#         json.dump(accuracies, file)
#         file.write("\n")
#         json.dump(mean_acc, file)
#         if extra is not None:
#             file.write("\n")
#             json.dump(extra, file)
#
#
# def search_best_models_all_folds(base_model_name: str, base_model_train_func: Callable[[np.ndarray, np.ndarray], Any],
#                                  data_by_fold: List[Dict]):
#     """
#     Search for each fold the best model
#     :param base_model_name:
#     :param base_model_train_func:
#     :param data_by_fold: List of dictionaries of each fold data split to x_train, x_test, y_train, y_test
#     :return: dictionary of the best models found. Keys are the model parameters.
#     """
#     best_results = dict()
#     for fold_num, fold in enumerate(data_by_fold):
#         print(f"Searching models fold num: {fold_num}")
#         best_acc, best_name, best_params, base_model_acc, base_model_params = get_best_model_for_fold(
#             **fold, base_model_name=base_model_name, base_model_train_func=base_model_train_func)
#         # Using OrderedDict is done to have consistency between all the params to allow managing the best models
#         # in a set without duplicates
#         best_params = OrderedDict(best_params)
#         best_results[frozenset([best_name, frozenset(best_params.values())])] = \
#             (best_acc, best_name, best_params, base_model_acc, base_model_params, fold_num)
#
#         print(f"\nFor fold: {fold_num} the best model is: {best_name}, with accuracy: {best_acc}, "
#               f"and parameters: {best_params}\n"
#               f"The Base Model ({base_model_name}) results: {base_model_acc}")
#     return best_results
#
#
# def evaluate_save_top_models(base_model_name: str, best_results: Dict, full_train_data: np.ndarray, labels: np.ndarray,
#                              minimum_improvement_required: float, data_by_fold: List[Dict], save_path: str):
#     """
#     Does K-fold cross validation on the models in best_results, if the model achieves better results than the base model
#     by minimum_improvement_required it will save the model
#     :param base_model_name:
#     :param best_results: Dictionary as returned from search_best_models_all_folds
#     :param full_train_data: All the train data for final fitting of the model
#     :param labels:
#     :param minimum_improvement_required: minimum improvement in accuracy compared to the base model
#     :param data_by_fold: List of dictionaries of each fold data split to x_train, x_test, y_train, y_test
#     :param save_path: model save directory
#     :return:
#     """
#     print("Evaluate best models")
#     for _, best_name, best_params, base_model_acc, base_model_params, fold_num in tqdm(best_results.values()):
#         mean_acc, models, accuracies = eval_model_on_all_folds(data_by_fold, best_params, best_name, )
#         lda_mean_acc, _, _ = eval_model_on_all_folds(data_by_fold, base_model_params, base_model_name)
#         print(f"\nMean acc for {best_name} is {mean_acc} and Base Model ({base_model_name}) mean acc: {lda_mean_acc}\n")
#         if mean_acc <= (base_model_acc + minimum_improvement_required):
#             print("Continue")
#         model_all = globals()[best_name](**best_params)
#         model_all.fit(full_train_data, labels)
#         models.append(model_all)
#         save_all_data(save_path, best_name, best_params, models, accuracies, mean_acc,
#                       f"fold: {fold_num} with lda acc: {base_model_acc} and params: {base_model_params}")
#
#
# def main(folds, minimum_improvement_required: float = 0.05, base_model_name: str = "LinearDiscriminantAnalysis",
#          base_model_train_func: Callable[[np.ndarray, np.ndarray], Any] = train_lda,
#          data_load_path: str = config.DATA_DIR, models_save_path: str = os.path.join(config.DATA_BASE, MODELS_DIR)):
#     f"""
#     Splits all data to folds, for each fold find the best model possible.
#     For each best model evaluate it using K-fold cross validation and compare to base model, if it improves on
#     base model accuracy by minimum_improvement_required it will save the model.
#     :param folds - how many folds to split the data
#     :param minimum_improvement_required: minimum improvement required compared to base model
#     :param base_model_name:
#     :param base_model_train_func:
#     :param data_load_path:
#     :param models_save_path:
#     :return:
#     """
#     data = load_data(config.TRAIN_FEATURES, folder=data_load_path)
#     labels = load_data(config.TRAINING_LABELS, folder=data_load_path)
#     labels = labels.reshape(-1,)
#     try:
#         trials_to_remove = load_data(config.TRIALS_TO_REMOVE, folder=data_load_path)
#         trials_to_remove = trials_to_remove.reshape(-1,)
#     except FileExistsError:
#         trials_to_remove = list()
#     data, labels = utils.remove_trials(data, labels, trials_to_remove)
#     skf = StratifiedKFold(n_splits=folds, shuffle=False)
#     split_data = list()
#     for train_index, test_index in skf.split(data, labels):
#         x_train, x_test = data[train_index], data[test_index]
#         y_train, y_test = labels[train_index], labels[test_index]
#         split_data.append({
#             "x_train": x_train,
#             "x_test": x_test,
#             "y_train": y_train,
#             "y_test": y_test
#         })
#
#     best_results = search_best_models_all_folds(base_model_name, base_model_train_func, split_data)
#     evaluate_save_top_models(base_model_name, best_results, data, labels, minimum_improvement_required,
#                              split_data, models_save_path)
