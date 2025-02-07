from src.optimization.genetic_algorithm.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters
import random
import numpy as np
import torch

class BridgeMovementSupplier(GAMovementsSupplier):
  def __init__(self, ga_params: GAParameters):
    super().__init__(ga_params)
    

  def create_individual(self):
   
    return torch.bernoulli(torch.full((self.ga_params.n_genes,), self.ga_params.proportion_rate)).int()

  # def create_individual(self):
   
  #   return torch.randint(0, 2, (self.ga_params.n_genes,))
  
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

  # def select(self, poblacion):
    
  #   seleccionados = sorted(poblacion, key=lambda x: x[0])
  #   padres_seleccionados = []
  #   cont = 0
  #   for tupla in seleccionados:
  #     if cont == 2:
  #         break
  #     padre = tupla[1]
  #     fitne = tupla[0]
  #     print(f"Parent {cont+1} {fitne}")
  #     padres_seleccionados.append(padre)
  #     cont += 1
      
        
  #   return padres_seleccionados
    
  def crossing(self, padre1, padre2):
    puntoCruce = random.randint(1, len(padre1) - 1)
    hijo1 = torch.cat((padre1[:puntoCruce], padre2[puntoCruce:]))
    hijo2 = torch.cat((padre2[:puntoCruce], padre1[puntoCruce:]))

    return hijo1, hijo2
  
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
