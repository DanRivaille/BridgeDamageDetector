from abc import ABC, abstractmethod
from random import random, sample

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
  def select(self, population, fitness: np.ndarray[float]):
    """
    Creates a new population applying some selecting strategy
    """
    pass

  @abstractmethod
  def make_children(self, father_1, father_2):
    """
    Creates two individuals applying some crossing strategy considering the parents' information
    """
    pass

  def crossing(self, population):
    """
    Creates a new population applying some crossing strategy
    """
    obj_proportion = (round(self.ga_params.n_genes * self.ga_params.proportion) / self.ga_params.n_genes) 


    for i in range(1, self.ga_params.population_size, 2):
      rand = random()

      if rand < self.ga_params.p_cross:
        population[i - 1], population[i] = self.make_children(population[i - 1], population[i])

        if(self.ga_params.is_prop):
          population[i - 1] = validate_prop(obj_proportion,population[i - 1], self.ga_params.n_genes)
          population[i] = validate_prop(obj_proportion,population[i], self.ga_params.n_genes)


    return population

  @abstractmethod
  def make_mutation(self, individual):
    """
    Creates a new individual applying some mutation strategy
    """
    pass

  def mutate(self, population):
    """
    Creates a new population applying some mutation strategy
    """
    for i in range(self.ga_params.population_size):
      rand = random()

      if rand < self.ga_params.p_mutate:
        population[i] = self.make_mutation(population[i])

    return population

  
def get_prop(solution, n_genes):
  cont1 = sum(solution)

  return cont1 / n_genes

def validate_prop(obj, solution, n_genes):
  prop = get_prop(solution,n_genes)
  current_ones = sum(solution)
  target_ones = round(n_genes * 0.8)

  if prop < obj:
    print(f"Proporcion no valida. Prop = {prop}")
    zero_indices = [i for i, bit in enumerate(solution) if bit == 0]
    to_convert = target_ones - current_ones

    to_flip = sample(zero_indices, to_convert)
    for idx in to_flip:
        solution[idx] = 1
    print(f"Actual prop = {get_prop(solution, n_genes)}")

  return solution
  
    






