import random

import numpy as np

from src.optimization.genetic_algorithm.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters


class KNPGAMovementSupplier(GAMovementsSupplier):
  def __init__(self, ga_params: GAParameters, k_select_param=3):
    super().__init__(ga_params)
    self.__k_select_param: int = k_select_param

  def create_individual(self):

    num_ones = round(self.ga_params.n_genes * self.ga_params.proportion)
    num_zeros = self.ga_params.n_genes - num_ones

    individual = [1] * num_ones + [0] * num_zeros

    random.shuffle(individual)

    return individual

  def select(self, population, fitness: np.ndarray[float]):
    winners = []

    for i in range(self.ga_params.population_size):
      index_participants = random.sample(range(self.ga_params.population_size), self.__k_select_param)
      winner_index = KNPGAMovementSupplier.__get_max_index(fitness, index_participants)
      winners.append(population[winner_index].copy())
      
    return winners

  def make_children(self, father_1, father_2):
    
    crossover_point = random.randint(0, self.ga_params.n_genes - 1)

    child1 = father_1[0:crossover_point] + father_2[crossover_point:]
    child2 = father_2[0:crossover_point] + father_1[crossover_point:]

    return child1, child2

  def make_mutation(self, individual):

    for i in range(len(individual)):
      prob = random.random()
      if prob <= self.ga_params.p_mutate and individual[i] == 0:
        individual[i] = 1
      elif prob <= self.ga_params.p_mutate and individual[i] == 1:
        individual[i] = 0
        
    return individual
    

  @staticmethod
  def __get_max_index(fitness, index_participants):
    index_max = index_participants[0]
    best_fitness = fitness[index_participants[0]]

    for index in index_participants:
      if fitness[index] > best_fitness:
        index_max = index
        best_fitness = fitness[index]

    return index_max
