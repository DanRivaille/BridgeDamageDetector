import numpy as np
import time

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from torch import nn, no_grad
from torch.utils.data import DataLoader

from src.damage_detector.Results import Results
from src.damage_detector.ConfigParams import ConfigParams


class AnomalyDetector:
    """
    A class that uses a trained MLModel to detect anomalies in given datasets.
    """
    def __init__(self, model: nn.Module, config: ConfigParams, device: str):
        """
        Initializes an instance of AnomalyDetector.
        @param model The trained MLModel used for anomaly detection.
        @param config The configuration parameters for the anomaly detection process.
        """
        self.__trained_model = model
        self.__config = config
        self.__device = device

        self.__macroseq_length: int = config.get_params('test_params')['macroseq_length']

    def predict(self, model_trained: nn.Module, data_to_predict: DataLoader) -> np.ndarray:
        feature_vector = []
        criterion = nn.MSELoss(reduction='none')

        with no_grad():
            for data_batch in data_to_predict:
                signals = data_batch.to(self.__device)
                output = model_trained(signals)

                current_feature_value = np.mean(criterion(output, signals.data).cpu().numpy(), axis=1)
                feature_vector.append(current_feature_value)

        return np.concatenate(feature_vector).flatten()

    def detect_damage(self, damaged_dataloader: DataLoader, healthy_dataloader: DataLoader) -> Results:
        """
        Detects features damaged in the provided datasets and returns the Results.
        @param damaged_dataloader DataLoader or numpy array containing damaged data.
        @param healthy_dataloader DataLoader or numpy array containing healthy data.
        """
        features_damaged = self.predict(self.__trained_model, damaged_dataloader)
        features_healthy = self.predict(self.__trained_model, healthy_dataloader)

        print(features_damaged.shape, features_damaged.mean())
        print(features_healthy.shape, features_healthy.mean())

        damage_detection_results = self.__find_best_thresholds(features_damaged, features_healthy)
        return Results(*damage_detection_results)

    def __detect_damage(self, feature_vector: np.ndarray, feature_threshold: float,
                        macroseq_threshold: float):
        """
        Detects damage using the specified feature and macro-sequence thresholds.
        @param feature_vector The feature vector to be evaluated for damage.
        @param feature_threshold The threshold for the feature vector.
        @param macroseq_threshold The threshold for macro-sequences.
        """
        # Se divide el vector de caracteristicas en macro-secuencias
        macroseq_feature_vector = self.__split_in_macrosequences(feature_vector)

        # Se calcula cuales secuencias dentro de cada macro-secuencia supera el umbral pre-establecido
        labels_vector = AnomalyDetector.__evaluate_thresholds(macroseq_feature_vector, feature_threshold)

        # Se calcula la proporcion de secuencias identificadas como dañadas dentro de cada macro-secuencia
        macrosequences_labels_vector = np.sum(labels_vector, axis=1) / self.__macroseq_length

        # Se etiqueta cada macro-secuencia dependiendo de si supera el umbral pre-establecido
        return AnomalyDetector.__evaluate_thresholds(macrosequences_labels_vector, macroseq_threshold)

    def __split_in_macrosequences(self, labels: np.ndarray) -> np.ndarray:
        """
        Split the vector into macro-sequences.
        @param labels The vector of labels to be split into macro-sequences.
        """
        n_samples = labels.shape[0]
        samples_to_consider = n_samples - (n_samples % self.__macroseq_length)
        return labels[:samples_to_consider].reshape((-1, self.__macroseq_length))

    @staticmethod
    def __evaluate_thresholds(feature_vector: np.ndarray, threshold: float) -> np.ndarray:
        """
        Evaluates which sequences or macro-sequences exceed the established threshold and labels them accordingly.
        @param feature_vector Vector of sequences or macro-sequences.
        @param threshold Threshold to evaluate.
        """
        return (feature_vector > threshold).astype(int)

    def __find_best_thresholds(self, feature_test_vector: np.ndarray, feature_valid_vector: np.ndarray) -> tuple:
        """
        Finds the best thresholds for features and macro-sequences based on the pre-set range in the configuration parameters.
        @param feature_test_vector The feature vector for testing the thresholds.
        @param feature_valid_vector The feature vector for validation the thresholds.
        """
        test_params = self.__config.get_params('test_params')
        feature_threshold_list = np.linspace(test_params['min_feature_threshold'], test_params['max_feature_threshold'],
                                             test_params['number_feature_thresholds_to_try'])
        macroseq_threshold_list = np.linspace(test_params['min_macroseq_threshold'], test_params['max_macroseq_threshold'],
                                             test_params['number_macroseq_thresholds_to_try'])

        max_auc = -1
        max_f1 = -1.0
        best_f_t = -1
        best_m_t = -1
        cmatrix = None
        start_time = time.time()

        for f_t in feature_threshold_list:
            for m_t in macroseq_threshold_list:
                damage_predicted = self.__detect_damage(feature_test_vector, f_t, m_t)
                health_predicted = self.__detect_damage(feature_valid_vector, f_t, m_t)

                true_labels = (np.concatenate((np.ones(damage_predicted.shape[0], dtype=int),
                                               np.zeros(health_predicted.shape[0], dtype=int)))).reshape((-1, 1))
                predicted_labels = (np.concatenate((damage_predicted, health_predicted))).reshape((-1, 1))

                auc_score = roc_auc_score(true_labels, predicted_labels)
                f1 = f1_score(true_labels, predicted_labels)

                if f1 > max_f1:
                    max_f1 = f1
                    max_auc = auc_score
                    best_f_t = f_t
                    best_m_t = m_t
                    cmatrix = confusion_matrix(true_labels, predicted_labels)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f'Best values: F_t: {best_f_t} - M_t: {best_m_t} - Max AUC score: {max_auc} - Max F1 score: {max_f1}')
        return best_f_t, best_m_t, max_f1, max_auc, elapsed_time, cmatrix
