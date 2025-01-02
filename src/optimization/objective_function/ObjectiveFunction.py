from abc import ABC, abstractmethod


class ObjectiveFunction(ABC):
  def __init__(self, is_minimization: bool):
    self.__is_minimization: bool = is_minimization

  def is_minimization(self) -> bool:
    return self.__is_minimization

  def compare_objective_values(self, fitness_value_1: float, fitness_value_2: float) -> int:
    """
    Compares two fitness values and determines if fitness_value_1 is better than fitness_value_2 (it will depend
    on whether we are maximizing or minimizing).

    If fitness_value_1 is better than fitness_value_2 it will return 1, otherwise 0 or -1 (0 if both values are equals,
    and -1 if fitness_value_2 is better than fitness_value_1).
    """
    if self.__is_minimization:
      return ObjectiveFunction.__compute_compared_value(fitness_value_2, fitness_value_1)
    else:
      return ObjectiveFunction.__compute_compared_value(fitness_value_1, fitness_value_2)

  @staticmethod
  def __compute_compared_value(value_1: float, value_2: float) -> int:
    """
    compares two fitness values, if the first is larger it returns 1, if the second is larger it returns -1,
    if they are equal it returns 0.
    """
    if value_1 > value_2:
      return 1
    elif value_1 < value_2:
      return -1
    else:
      return 0

  @abstractmethod
  def evaluate(self, solution) -> float:
    pass
