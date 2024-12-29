from enum import Enum


class DatasetType(Enum):
    TRAIN_DATA = 'train_data'
    TEST_DATA = 'test_data'
    VALIDATION_TRAIN_DATA = 'validation_train_data'
    VALIDATION_TEST_DATA = 'validation_test_data'
