from json import dump

import numpy as np


class Results:
    """
    A class for saving the results of model execution.
    """
    def __init__(self,
                 feature_threshold: float,
                 macroseq_threshold: float,
                 max_f1: float,
                 max_auc: float,
                 execution_time: float,
                 confusion_matrix: np.ndarray):
        """
        Initializes an instance of Results.
        @param feature_threshold The threshold to consider whether a feature is damaged.
        @param macroseq_threshold The threshold to consider whether the macrosequence is damaged.
        @param max_f1 The maximum F1 score achieved.
        @param max_auc The maximum Area Under the Curve (AUC) achieved.
        @param damaged_features An array with the features considered as damaged.
        @param healthy_features An array with the features considered as healthy.
        @param execution_time The execution time of the model's anomaly detection process.
        """
        self.__feature_threshold = feature_threshold
        self.__macroseq_threshold = macroseq_threshold
        self.__max_f1 = max_f1
        self.__max_auc = max_auc
        self.__confusion_matrix = confusion_matrix
        self.__execution_time = execution_time

    def to_json(self) -> dict:
        """
        Returns a dictionary for a .json file with the results.
        """
        return {
            "feature_threshold": self.__feature_threshold,
            "macroseq_threshold": self.__macroseq_threshold,
            "max_f1": self.__max_f1,
            "max_auc": self.__max_auc,
            "confusion_matrix": self.__confusion_matrix.tolist(),
            "execution_time": self.__execution_time
        }

    def save(self, path: str):
        """
        Saves the results to a .json file.
        @param path The file path where the results will be saved.
        """
        with open(path, "w") as json_results_file:
            dump(self.to_json(), json_results_file)
