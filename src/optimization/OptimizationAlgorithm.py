from abc import ABC, abstractmethod


class OptimizationAlgorithm(ABC):
  @abstractmethod
  def set_params(self, **params):
    pass

  @abstractmethod
  def run(self, **kwargs):
    pass
