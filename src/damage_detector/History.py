from json import dump


class History:
    """
    A class for saving the training history data of model execution.
    """
    def __init__(self,
                 train_loss: list,
                 valid_loss: list,
                 learning_rate,
                 execution_time: float):
        """
        Initializes an instance of History.
        @param train_loss List of training losses for the model.
        @param valid_loss List of validation losses for the model.
        @param learning_rate List of learning rates during the training of the model.
        @param execution_time The execution time of the model training process.
        """
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.learning_rate = learning_rate
        self.execution_time = execution_time

    def to_json(self) -> dict:
        """
        Returns a dictionary for a .json file with the training history.
        """
        entries = {
            'train_loss': self.train_loss,
            'valid_loss': self.valid_loss,
            'learning_rate_updating': self.learning_rate,
            'execution_time': self.execution_time
        }

        return entries

    def save(self, path: str):
        """
        Saves the training history to a .json file.
        @param path The file path where the results will be saved.
        """
        with open(path, 'w') as history_file:
            dump(self.to_json(), history_file)
