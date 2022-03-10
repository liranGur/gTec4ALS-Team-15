import abc
from abc import ABC
from typing import List, Dict, Type

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from config import *


class BaseParamSearcher(ABC):

    def __init__(self, parameters: List[Dict], model):
        super(BaseParamSearcher, self).__init__()
        self.model = model
        self.parameters = parameters

    @abc.abstractmethod
    def param_search(self, data: np.ndarray, labels: np.ndarray) -> None:
        """
        Search for best Parameters
        :param data:
        :param labels:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_best_params(self) -> Dict:
        """
        return best params after search
        :return:
        """
        pass

    @abc.abstractmethod
    def predict(self, data: np.ndarray) -> np.ndarray:
        pass


class GridSearchParamSearcher(BaseParamSearcher):
    def __init__(self, parameters: List[Dict], model):
        super(GridSearchParamSearcher, self).__init__(parameters, model)
        self.grid_clf = GridSearchCV(model, parameters, scoring="accuracy", cv=GRID_SEARCH_CV)

    def param_search(self, data: np.ndarray, labels: np.ndarray):
        self.grid_clf.fit(data, labels)

    def get_best_params(self):
        return self.grid_clf.best_params_

    def predict(self, data: np.ndarray):
        return self.grid_clf.predict(data)


class CatBoostParamSearcher(BaseParamSearcher):

    def __init__(self, parameters: List[Dict], model):
        super(CatBoostParamSearcher, self).__init__(parameters, CatBoostClassifier())
        self.best_params = dict()
        self.model = model

    def param_search(self, data: np.ndarray, labels: np.ndarray) -> None:
        self.best_params = self.model.grid_search(self.parameters, X=data, y=labels,
                                                  cv=GRID_SEARCH_CV, verbose=False)['params']
        self.model = CatBoostClassifier(**self.best_params)
        self.model.fit(data, labels)

    def get_best_params(self) -> Dict:
        return self.best_params

    def predict(self, data: np.ndarray) -> np.ndarray:
        return self.model.predict(data)


def use_fake_params_if_needed(parameters):
    """
    This is for testing purposes only
    :param parameters: search parameters for model
    :return: Real parameters to search or only a single parameter for a fake quick grid search
    """
    if FAKE_PARAMS_FOR_TEST:
        return [p[0] for p in parameters]
    return parameters


class BaseClassifierSearch(ABC):

    name = None

    def __init__(self):
        self.grid_search: BaseParamSearcher = None
        self.create_params = self._create_params
        self.model_type = None
        self.model = None
        print(f"Init model: {self.name}")

    @abc.abstractmethod
    def search_parameters(self, x_train: np.ndarray, y_train: np.ndarray) -> BaseParamSearcher:
        pass

    def _base_search(self, parameters: List[List], data: np.ndarray, labels: np.ndarray,
                     searcher: Type[BaseParamSearcher] = GridSearchParamSearcher) -> BaseParamSearcher:
        print(f"Searching parameters for {self.name}")
        parameters = use_fake_params_if_needed(parameters)
        grid_search_params = self.create_params(*parameters)
        self.grid_search = searcher(grid_search_params, self.model_type())
        self.grid_search.param_search(data, labels)
        self.model = self.grid_search
        print(f"Finished Searching for {self.name}")
        return self.grid_search

    def model_test_acc(self, x_test, y_test, **kwargs) -> float:
        preds = self.model.predict(x_test)
        return accuracy_score(y_test, preds)

    def get_best_params(self) -> Dict:
        return self.grid_search.get_best_params()

    def create_model_and_train(self, parameters, x_train, y_train, **kwargs):
        self.model = self.model_type(**parameters)
        self.model.fit(x_train, y_train)


class RandomForestSearch(BaseClassifierSearch):

    name = "RandomForest"

    def __init__(self):
        super(RandomForestSearch, self).__init__()
        self.model_type = RandomForestClassifier

    def search_parameters(self, x_train: np.ndarray, y_train: np.ndarray) -> BaseParamSearcher:
        parameters = [[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000],
                      [10, 25, 50, 100, 125, 150, 300],
                      ["gini"]
                      ]
        # parameters = [[i for i in range(500, 2000, 250)],
        #               [i for i in range(5, 40, 5)] + [i for i in range(50, 600, 50)],
        #               ["gini", "entropy"]
        #               ]
        #
        # if RF_PARAMS_TO_USE != NO_PARAMETERS_SPLIT:
        #     options = [[parameters[0], parameters[1][:len(parameters[1])//2], parameters[2]],
        #                [parameters[0], parameters[1][len(parameters[1])//2:], parameters[2]]]
        #     results = self._base_search(options[RF_PARAMS_TO_USE], x_train, y_train)
        # else:
        #     results = self._base_search(parameters, x_train, y_train)

        results = self._base_search(parameters, x_train, y_train)
        return results

    @staticmethod
    def _create_params(n_estimators, max_depth, criterion, n_jobs=4):
        return {k: [v] if type(v) is not list else v for k, v in locals().items()}


class SVCSearch(BaseClassifierSearch):

    name = "SVC"

    def __init__(self):
        super(SVCSearch, self).__init__()
        self.model_type = SVC

    def search_parameters(self, x_train: np.ndarray, y_train: np.ndarray) -> BaseParamSearcher:
        parameters = [[0.01, 0.5, 0.9, 1, 1.1, 1.5],
                      ["rbf", "sigmoid"],
                      [2, 4, 5, 6],
                      ]
        # parameters = [[0.01, 0.5, 0.9, 1, 1.1, 1.5],
        #               ["linear", "poly", "rbf", "sigmoid"],
        #               [2, 3, 4, 5, 6],
        #               [True, False],
        #               ["scale", "auto"],
        #               [0.0, 0.1, 0.01, 0.001, 0.2, 0.02, 0.3, 0.4]
        #               ]
        return self._base_search(parameters, x_train, y_train)

    @staticmethod
    def _create_params(C, kernel, degree):   # , shrinking, gamma, coef0):
        return {k: [v] if type(v) is not list else v for k, v in locals().items()}


class SGDSearch(BaseClassifierSearch):

    name = "SGD"

    def __init__(self):
        super(SGDSearch, self).__init__()
        self.model_type = SGDClassifier

    def search_parameters(self, x_train: np.ndarray, y_train: np.ndarray) -> BaseParamSearcher:
        parameters = [["hinge", "log", "modified_huber", "squared_hinge", "perceptron"],
                      ["l2", "l1", "elasticnet"],
                      [1e-3, 5e-3, 1e-4, 5e-4, 1e-5, 5e-5]
                      ]
        return self._base_search(parameters, x_train, y_train)

    @staticmethod
    def _create_params(loss, penalty, alpha, n_jobs=4):
        return {k: [v] if type(v) is not list else v for k, v in locals().items()}


class LDASearch(BaseClassifierSearch):

    name = "LDA"

    def __init__(self):
        super(LDASearch, self).__init__()
        self.model_type = LinearDiscriminantAnalysis

    def search_parameters(self, x_train: np.ndarray, y_train: np.ndarray) -> BaseParamSearcher:
        parameters = [["svd", "lsqr", "eigen"],
                      [None, "auto", 1e-3, 1e-2, 0.1, 0.3, 0.5, 0.7, 1]
                      ]
        return self._base_search(parameters, x_train, y_train)

    @staticmethod
    def _create_params(solver, shrinkage):
        return {k: [v] if type(v) is not list else v for k, v in locals().items()}


class XgBoostSearch(BaseClassifierSearch):

    name = "XGBoost"

    def __init__(self):
        super(XgBoostSearch, self).__init__()
        self.model_type = XGBClassifier

    def search_parameters(self, x_train: np.ndarray, y_train: np.ndarray) -> BaseParamSearcher:
        parameters = [[2000, 2250, 2500, 3000],
                      [0.25, 0.3, 0.35, 0.4, 0.45],
                      [8, 10, 12, 15],
                      [0.45, 0.5, 0.6, 0.7, 0.8, 0.85]
                      ]
        # parameters = [[i for i in range(1000, 3000, 250)],
        #               [0.1 ** i for i in range(1, 6, 1)] + [0.5 ** i for i in range(1, 4, 1)],
        #               ["gbtree", "gblinear", "dart", "hist"],
        #               ["exact", "approx", "hist"],
        #               # [None] + [i for i in range(5, 30, 5)] + [i for i in range(50, 600, 50)],
        #               # [1 - 0.5 * i for i in range(0, 10)],
        #               # [0.1 * i for i in range(0, 11)] + [1.5, 2, 2.5, 3]
        #               ]
        # max_depth_opts = [[i for i in range(5, 15, 5)],
        #                   [i for i in range(15, 30, 5)],
        #                   [i for i in range(50, 300, 50)],
        #                   [i for i in range(300, 600, 50)]]
        # max_depth_opts = [[None] + x for x in max_depth_opts]
        # gamma_opts = [[1 - 0.5 * i for i in range(0, 5)],
        #               [1 - 0.5 * i for i in range(5, 10)]]
        # subsample_opts = [[0.1 * i for i in range(0, 8)],
        #                   [0.1 * i for i in range(8, 11)] + [1.5, 2, 2.5, 3]]
        #
        # parameters.append(max_depth_opts[XGBOOST_PARAMS_TO_USE])
        # parameters.append(gamma_opts[XGBOOST_PARAMS_TO_USE//2])
        # parameters.append(subsample_opts[XGBOOST_PARAMS_TO_USE//2])

        return self._base_search(parameters, x_train, y_train)

    @staticmethod
    def _create_params(n_estimators, learning_rate, max_depth, subsample, verbosity=0, n_jobs=4):
    # def _create_params(n_estimators, learning_rate, booster, tree_method,
    #                    max_depth, gamma, subsample,
    #                    verbosity=0, n_jobs=4):
        return {k: [v] if type(v) is not list else v for k, v in locals().items()}


class CatBoostSearch(BaseClassifierSearch):

    name = "CatBoost"

    def __init__(self):
        super(CatBoostSearch, self).__init__()
        self.model_type = CatBoostClassifier

    def search_parameters(self, x_train: np.ndarray, y_train: np.ndarray) -> BaseParamSearcher:
        # parameters = [[i for i in range(500, 1500, 200)],
        #               [0.1 ** i for i in range(1, 6, 1)] + [0.5 ** i for i in range(1, 4, 1)],
        #               [i for i in range(15)],
        #               [1 - 0.1*i for i in range(0, 10)]
        #               # [i for i in range(1, 128, 10)]
        #               ]
        #
        # border_count_opts = [[i for i in range(1, 128, 10)], [i for i in range(128, 256, 10)]]
        # parameters.append(border_count_opts[CAT_BOOST_PARAMS_TO_USE])

        parameters = [[1000, 2000, 3000],
                      [0.3, 0.5, 0.7, 0.9],
                      [10, 50, 150, 400, 700],
                      ]

        return self._base_search(parameters, x_train, y_train, CatBoostParamSearcher)

    @staticmethod
    def _create_params(iterations, learning_rate, depth,  # , l2_leaf_reg, border_count,
                       logging_level="Silent"):
        return {k: [v] if type(v) is not list else v for k, v in locals().items()}
