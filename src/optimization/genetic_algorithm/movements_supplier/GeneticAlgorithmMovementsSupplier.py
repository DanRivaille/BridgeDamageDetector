from abc import ABC, abstractmethod

import numpy as np

from src.optimization.objective_function.ObjectiveFunction import ObjectiveFunction
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters


class GAMovementsSupplier(ABC):
  def __init__(self, ga_params: GAParameters):
    self.ga_params: GAParameters = ga_params

  @staticmethod
  def get_best(objective_function: ObjectiveFunction, population, fitness: np.ndarray[float]) -> tuple:
    """
    Gets the best individual (and its fitness) in the population
    """
    if objective_function.is_minimization():
      best_fitness_index = np.argmin(fitness)
    else:
      best_fitness_index = np.argmax(fitness)

    return population[best_fitness_index], fitness[best_fitness_index]

  @staticmethod
  def compute_population_fitness(objective_function: ObjectiveFunction, population) -> np.ndarray[float]:
    """
    Computes the fitness of each individual of the population
    """
    fitness = np.array([objective_function.evaluate(individual) for individual in population], dtype=float)
    return fitness

  def create_population(self):
    """
    Creates a new population considering population_size of individuals
    """
    return [self.create_individual() for i in range(self.ga_params.population_size)]

  @abstractmethod
  def create_individual(self):
    """
    Creates a new individual
    """
    pass

  @abstractmethod
  def select(self, population):
    """
    Creates a new population applying some selecting strategy
    """
    pass

  @abstractmethod
  def crossing(self, father_1, father_2):
    """
    Creates two individuals applying some crossing strategy considering the parents' information
    """
    pass

  @abstractmethod
  def mutate(self, individual):
    """
    Creates a new individual applying some mutation strategy
    """
    pass
