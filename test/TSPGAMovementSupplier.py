import random

import numpy as np

from src.optimization.genetic_algorithm.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters


class TSPGAMovementSupplier(GAMovementsSupplier):
  def __init__(self, ga_params: GAParameters, k_select_param=3):
    super().__init__(ga_params)
    self.__k_select_param: int = k_select_param

  def create_individual(self):
    return random.sample(range(self.ga_params.n_genes), self.ga_params.n_genes)

  def select(self, population, fitness: np.ndarray[float]):
    winners = []

    for i in range(self.ga_params.population_size):
      index_participants = random.sample(range(self.ga_params.population_size), self.__k_select_param)

      winner_index = TSPGAMovementSupplier.__get_min_index(fitness, index_participants)
      winners.append(population[winner_index])

    return winners

  def make_children(self, father_1, father_2):
    n_genes = self.ga_params.n_genes
    # Se crean los hijos sin genes todavia
    children_1 = [-1] * n_genes
    children_2 = [-1] * n_genes

    # Cantidad de genes que van a ser heredados a los hijos
    cant_genes_inherited = 4

    # Se selecciona un indice inicial al azar, del cual se empezara a copiar los genes
    initial_index = random.randint(0, n_genes - 1)

    # Se copian los genes heredados
    children_1 = copy_genes(father_1, children_1, initial_index, cant_genes_inherited, n_genes)
    children_2 = copy_genes(father_2, children_2, initial_index, cant_genes_inherited, n_genes)

    # Se inicializan los indices
    index = (initial_index + cant_genes_inherited) % n_genes
    index_children_1 = index
    index_children_2 = index

    # Mientras no se de la vuelta completa a los padres, se van copiando genes
    while True:

      # Si el gen actual del padre no esta en el hijo, se copia
      if father_1[index] not in children_2:
        children_2[index_children_2] = father_1[index]
        index_children_2 = (index_children_2 + 1) % n_genes

      if father_2[index] not in children_1:
        children_1[index_children_1] = father_2[index]
        index_children_1 = (index_children_1 + 1) % n_genes

      index = (index + 1) % n_genes

      if index == ((initial_index + cant_genes_inherited) % n_genes):
        break

    return children_1, children_2

  def make_mutation(self, individual):
    indexes = random.sample(range(self.ga_params.n_genes), 2)
    individual[indexes[0]], individual[indexes[1]] = individual[indexes[1]], individual[indexes[0]]

    return individual

  @staticmethod
  def __get_min_index(fitness, index_participants):
    index_min = index_participants[0]
    best_fitness = fitness[index_participants[0]]

    for index in index_participants:
      if fitness[index] < best_fitness:
        index_min = index
        best_fitness = fitness[index]

    return index_min


def copy_genes(individual_origin, individual_destiny, initial, quantity, n_genes):
  index = initial
  while quantity > 0:
    individual_destiny[index] = individual_origin[index]

    index = (index + 1) % n_genes
    quantity -= 1

  return individual_destiny
