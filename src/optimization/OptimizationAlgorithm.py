from abc import ABC, abstractmethod


class OptimizationAlgorithm(ABC):
  @abstractmethod
  def set_params(self, **kargs):
    pass
