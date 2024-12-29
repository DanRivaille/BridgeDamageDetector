from abc import ABC, abstractmethod

from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters

import numpy as np


class GAMovementsSupplier(ABC):
  def __init__(self, ga_params: GAParameters):
    self.__ga_params: GAParameters = ga_params

  def get_best(self, population, fitness: np.ndarray[float]) -> tuple:
    """
    Gets the best individual (and its fitness) in the population
    """
    # TODO: generalize to use argmax or argmin
    best_fitness_index = np.argmin(fitness)

    return population[best_fitness_index], fitness[best_fitness_index]

  def compute_population_fitness(self, population) -> np.ndarray[float]:
    """
    Computes the fitness of each individual of the population
    """
    fitness = np.array([self.compute_individual_fitness(individual) for individual in population], dtype=float)
    return fitness

  def create_population(self):
    """
    Creates a new population considering population_size of individuals
    """
    return [self.create_individual() for i in range(self.__ga_params.population_size)]

  @abstractmethod
  def create_individual(self):
    """
    Creates a new individual
    """
    pass

  @abstractmethod
  def compute_individual_fitness(self, individual) -> float:
    """
    Computes an individual's fitness
    """
    pass

  @abstractmethod
  def select(self, population, fitness: np.ndarray[float]):
    """
    Creates a new population applying some selecting strategy
    """
    pass

  @abstractmethod
  def crossing(self, population):
    """
    Creates a new population applying some crossing strategy
    """
    pass

  @abstractmethod
  def mutate(self, population):
    """
    Creates a new population applying some mutation strategy
    """
    pass
