from src.optimization.genetic_algorithm.movements_supplier.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters
import random
import torch


class BridgeMovementSupplier(GAMovementsSupplier):
  def __init__(self, ga_params: GAParameters):
    super().__init__(ga_params)
    self.__FITNESS_INDEX = 0
    self.__INDIVIDUAL_INDEX = 1

  def create_individual(self):
    return torch.bernoulli(torch.full((self.ga_params.n_genes,), self.ga_params.proportion_rate)).int()

  def select(self, population):
    """
    Selection using tournament-based strategy.
    """
    parents = []
    random.shuffle(population)

    # Tournament between first and second
    if population[0][self.__FITNESS_INDEX] > population[1][self.__FITNESS_INDEX]:
      parents.append(population[0][self.__INDIVIDUAL_INDEX])
    else:
      parents.append(population[1][self.__INDIVIDUAL_INDEX])

    # Tournament between third and fourth
    if population[2][self.__FITNESS_INDEX] > population[3][self.__FITNESS_INDEX]:
      parents.append(population[2][self.__INDIVIDUAL_INDEX])
    else:
      parents.append(population[3][self.__INDIVIDUAL_INDEX])

    return parents

  def crossing(self, father_1, father_2):
    cross_point = random.randint(1, len(father_1) - 1)
    child_1 = torch.cat((father_1[:cross_point], father_2[cross_point:]))
    child_2 = torch.cat((father_2[:cross_point], father_1[cross_point:]))

    return child_1, child_2

  def mutate(self, genome):
    for i in range(len(genome)):
      prob = random.random()
      if prob <= self.ga_params.p_mutate and genome[i] == 0:
        genome[i] = 1
      elif prob <= self.ga_params.p_mutate and genome[i] == 1:
        genome[i] = 0

    return genome

  @staticmethod
  def __get_max_index(fitness, index_participants):
    index_max = index_participants[0]
    best_fitness = fitness[index_participants[0]]

    for index in index_participants:
      if fitness[index] > best_fitness:
        index_max = index
        best_fitness = fitness[index]

    return index_max
