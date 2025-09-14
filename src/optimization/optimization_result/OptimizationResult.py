from abc import ABC, abstractmethod
import json


class OptimizationResult(ABC):
  @abstractmethod
  def to_json(self) -> dict:
    """
    Returns a dict with the optimization result info
    """
    pass

  def save(self, file_path):
    """
    Saves the training history to a .json file.
    @param file_path The file path where the results will be saved.
    """
    with open(file_path, 'w') as optimization_result_file:
      json.dump(self.to_json(), optimization_result_file)
