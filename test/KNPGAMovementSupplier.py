import random

import numpy as np

from src.optimization.genetic_algorithm.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters


class KNPGAMovementSupplier(GAMovementsSupplier):
  def __init__(self, ga_params: GAParameters, k_select_param=3):
    super().__init__(ga_params)
    self.__k_select_param: int = k_select_param

  def create_individual(self):
    return [random.randint(0, 1) for _ in range(self.ga_params.n_genes)]
    

  def select(self, population, fitness: np.ndarray[float]):
    winners = []

    for i in range(self.ga_params.population_size):
      index_participants = random.sample(range(self.ga_params.population_size), self.__k_select_param)
      winner_index = KNPGAMovementSupplier.__get_max_index(fitness, index_participants)
      winners.append(population[winner_index].copy())
      
    return winners

  def make_children(self, father_1, father_2):
    """
    Implementacion momentanea de prueba
    """

    
    n_genes = self.ga_params.n_genes
    children_1 = father_1.copy()
    children_2 = father_2.copy()

    # Seleccionar un punto de cruce aleatorio
    crossover_point = random.randint(1, n_genes - 1)
    
    # Intercambiar los genes después del punto de cruce
    children_1[crossover_point:] = father_2[crossover_point:]
    children_2[crossover_point:] = father_1[crossover_point:]

    return children_1, children_2

  def make_mutation(self, individual):
    """
    Para problemas binarios, la mutación simplemente cambia un bit
    """
    mutation_point = random.randint(0, self.ga_params.n_genes - 1)
    individual[mutation_point] = 1 - individual[mutation_point]  # Cambia 0 por 1 o 1 por 0
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


def copy_genes(individual_origin, individual_destiny, initial, quantity, n_genes):
  index = initial
  while quantity > 0:
    individual_destiny[index] = individual_origin[index]

    index = (index + 1) % n_genes
    quantity -= 1

  return individual_destiny
