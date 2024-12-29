from abc import ABC, abstractmethod

from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters


class GAMovementsSupplier(ABC):
  def __init__(self, ga_params: GAParameters):
    self.__ga_params: GAParameters = ga_params

  @abstractmethod
  def create_population(self):
    pass

  @abstractmethod
  def calc_fitness_population(self, population):
    pass

  @abstractmethod
  def get_best(self, population, fitness) -> tuple:
    pass

  @abstractmethod
  def select(self, population, fitness):
    pass

  @abstractmethod
  def crossing(self, population):
    pass

  @abstractmethod
  def mutate(self, population):
    pass
