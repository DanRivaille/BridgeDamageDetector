from abc import ABC, abstractmethod


class OptimizationAlgorithm(ABC):

  @abstractmethod
  def run(self, **kwargs):
    pass
