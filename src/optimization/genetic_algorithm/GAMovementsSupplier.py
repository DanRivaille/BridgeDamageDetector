from abc import ABC, abstractmethod


class GeneticAlgorithmMovementsSupplier(ABC):
  @abstractmethod
  def create_population(self, n_genes, population_size):
    pass

  @abstractmethod
  def calc_fitness_population(self, population):
    pass

  @abstractmethod
  def get_best(self, population, fitness):
    pass

  @abstractmethod
  def select(self, population, fitness, n, population_size):
    pass

  @abstractmethod
  def crossing(self, population, population_size, p_cross):
    pass

  @abstractmethod
  def mutate(self, population, population_size, p_mutate):
    pass
