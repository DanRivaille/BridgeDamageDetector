import os
from datetime import datetime, timedelta

import numpy as np
from scipy.io import loadmat

from src.damage_detector.CommonPath import CommonPath
from src.damage_detector.ConfigParams import ConfigParams
from src.damage_detector.DatasetType import DatasetType


def stack_arrays(stacked_array: np.ndarray, new_array: np.ndarray) -> np.ndarray:
    """
    Stacks an array inside an array stack, if the array stack is empty then returns the new array.
    @param stacked_array Current array stack (may be empty).
    @param new_array A new array to enter the stack.
    """
    if stacked_array.shape[0] != 0:
        return np.vstack((stacked_array, new_array))
    else:
        return new_array


def load(config: ConfigParams, type_dataset: DatasetType) -> np.ndarray:
    """
    Loads the data of the Z24 dataset based on configuration parameters and dataset type.
    @param config Configuration parameters for the dataset loaded from a .json file (they are the first date and last date).
    @param type_dataset The type of dataset that the model will use (TRAIN, TEST, VALIDATION TRAIN o VALIDATION TEST).
    """
    data_params = config.get_params('data_params').get(type_dataset.value)
    first_date = datetime.strptime(data_params.get('first_date'), "%d/%m/%Y")
    last_date = datetime.strptime(data_params.get('last_date'), "%d/%m/%Y")
    sensor_number = config.get_params('data_params').get('sensor_number')

    total_hours = (last_date - first_date + timedelta(days=1)) // timedelta(hours=1)
    year = str(first_date.year)
    data = np.empty(0)

    for current_hour in range(total_hours):
        current_datetime = first_date + timedelta(hours=current_hour)

        folder_name = f'{str(current_datetime.month).zfill(2)}{str(current_datetime.day).zfill(2)}'
        filename = f'd_{year[2:]}_{current_datetime.month}_{current_datetime.day}_{current_datetime.hour}.mat'
        file_path = os.path.join(CommonPath.DATA_FOLDER.value, folder_name, filename)

        
        if os.path.isfile(file_path):
            
            data_mat = loadmat(file_path)
            new_data = data_mat['Data'][:, sensor_number]

            data = stack_arrays(data, new_data)
        
            

    return data
