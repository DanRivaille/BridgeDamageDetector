import os

import numpy as np
from torch.cuda import is_available
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler

from src.damage_detector.CommonPath import CommonPath
from src.damage_detector.ConfigParams import ConfigParams
from src.damage_detector.DatasetType import DatasetType
from src.damage_detector.Z24Dataset import load


def __get_device() -> str:
  """
    Gets the device string for TensorFlow operations, either GPU or CPU.
    """
  if is_available():
    current_device = 'cuda'
  else:
    current_device = 'cpu'

  print(current_device)
  return current_device


def split_in_sequences(data, sequence_length):
  sample_length, n_samples = data.shape
  sequence_samples_to_consider = (sample_length // sequence_length) * sequence_length
  return data.T[:, :sequence_samples_to_consider].reshape((-1, sequence_length)).T


def reshape_sequences(data: np.ndarray, new_length: int) -> np.ndarray:
  new_data = data.flatten()
  N_samples = new_data.shape[0]
  samples_to_consider = N_samples - (N_samples % new_length)

  return new_data[:samples_to_consider].reshape((-1, new_length))


def build_model_folder_path(model_identifier: str, config_identifier: str, folder_name: str):
  """
    Builds a directory path for a model and its configuration based on the provided identifiers
    @param model_identifier A string identifier for the current model.
    @param config_identifier A string identifier of the current model configuration.
    @param folder_name The folder name to save the runs.
    """
  model_folder = os.path.join(folder_name, f'{model_identifier}_cnf_{config_identifier}')
  return os.path.join(CommonPath.MODEL_PARAMETERS_FOLDER.value, model_folder)


def load_data(config_params: ConfigParams, is_train=True) -> tuple:
  sequences_length = config_params.get_params('global_variables').get('sequences_length')

  if is_train:
    dataset_type = DatasetType.TRAIN_DATA
    validation_dataset_type = DatasetType.VALIDATION_TRAIN_DATA
    type_data_str = 'Train'
  else:
    dataset_type = DatasetType.TEST_DATA
    validation_dataset_type = DatasetType.VALIDATION_TEST_DATA
    type_data_str = 'Test'

  # Load data
  data = load(config_params, dataset_type).T
  validation_data = load(config_params, validation_dataset_type).T
  print(f"{type_data_str} data shape (after the load): {data.shape}")
  print(f"Vali data shape (after the load): {validation_data.shape}")

  # Process the data
  if is_train:
    standardScaler = StandardScaler()
    maxAbsScaler = MaxAbsScaler()
    minMaxScaler = MinMaxScaler()
  else:
    standardScaler = joblib.load('/home/ivan.santos/repositories/BridgeDamageDetector/models_trained/standard_scaler.pkl')
    maxAbsScaler = joblib.load('/home/ivan.santos/repositories/BridgeDamageDetector/models_trained/max_abs_scaler.pkl')
    minMaxScaler = joblib.load('/home/ivan.santos/repositories/BridgeDamageDetector/models_trained/min_max_scaler.pkl')

  if is_train:
    standardScaler.fit(data)

  data = standardScaler.transform(data)
  validation_data = standardScaler.transform(validation_data)

  data = split_in_sequences(data, 1000)
  validation_data = split_in_sequences(validation_data, 1000)

  print(f"{type_data_str} data shape (after the sequence splitting): {data.shape}")
  print(f"Vali data shape (after the sequence splitting): {validation_data.shape}")

  if is_train:
    maxAbsScaler.fit(data)

  data = maxAbsScaler.transform(data).T
  validation_data = maxAbsScaler.transform(validation_data).T

  if is_train:
    minMaxScaler.fit(data.T)

  data = minMaxScaler.transform(data.T).T
  validation_data = minMaxScaler.transform(validation_data.T).T

  if is_train:
    joblib.dump(standardScaler, '/home/ivan.santos/repositories/BridgeDamageDetector/models_trained/standard_scaler.pkl')
    joblib.dump(maxAbsScaler, '/home/ivan.santos/repositories/BridgeDamageDetector/models_trained/max_abs_scaler.pkl')
    joblib.dump(minMaxScaler, '/home/ivan.santos/repositories/BridgeDamageDetector/models_trained/min_max_scaler.pkl')

  print(f"{type_data_str} data shape (after the scaling): {data.shape}")
  print(f"Vali data shape (after the scaling): {validation_data.shape}")

  data = reshape_sequences(data, sequences_length)
  validation_data = reshape_sequences(validation_data, sequences_length)

  # data = data.reshape((-1, sequences_length, n_features))
  # validation_data = validation_data.reshape((-1, sequences_length, n_features))

  print(f"{type_data_str} data shape (after the adding the feature): {data.shape}")
  print(f"Vali data shape (after the adding the feature): {validation_data.shape}")

  return data, validation_data
