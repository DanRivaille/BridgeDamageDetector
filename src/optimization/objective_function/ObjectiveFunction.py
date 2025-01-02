from abc import ABC, abstractmethod


class ObjectiveFunction(ABC):
  @abstractmethod
  def evaluate(self, solution) -> float:
    pass
