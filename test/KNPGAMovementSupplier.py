from src.optimization.genetic_algorithm.movements_supplier.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters
import random


class KNPGAMovementSupplier(GAMovementsSupplier):
  def __init__(self, ga_params: GAParameters):
    super().__init__(ga_params)
    

  def create_individual(self):
   
    return [random.choice([0, 1]) for _ in range(self.ga_params.n_genes)]

  def select(self, population):
    """
    Selection using tournament-based strategy.
    """
    parents = []
    random.shuffle(population)

    # Tournament between first and second
    if population[0][0] > population[1][0]:  
        parents.append(population[0][1])    
    else:
        parents.append(population[1][1])

    # Tournament between third and fourth
    if population[2][0] > population[3][0]:  
        parents.append(population[2][1])    
    else:
        parents.append(population[3][1])

    return parents
    
  def crossing(self, father_1, father_2):
    
    crossover_point = random.randint(0, self.ga_params.n_genes - 1)

    child1 = father_1[0:crossover_point] + father_2[crossover_point:]
    child2 = father_2[0:crossover_point] + father_1[crossover_point:]

    return child1, child2
  
  def mutate(self, genome, prob_mutacion):
    
    for i in range(len(genome)):
        prob = random.random()
        if prob <= prob_mutacion and genome[i] == 0:
            genome[i] = 1
        elif prob <= prob_mutacion and genome[i] == 1:
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
