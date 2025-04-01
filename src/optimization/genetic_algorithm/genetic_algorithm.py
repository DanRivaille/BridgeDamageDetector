import logging

from src.optimization.optimization_algorithm import OptimizationAlgorithm
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters
from src.optimization.genetic_algorithm.movements_supplier.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier
from src.optimization.objective_function.ObjectiveFunction import ObjectiveFunction
from collections import Counter


def validate_termination(fitness):
  fitness_count = Counter(fitness)

  total_gen = len(fitness)
  limit = total_gen * 0.95

  return any(value >= limit for value in fitness_count.values())


class GeneticAlgorithm(OptimizationAlgorithm):
  def __init__(self,
               ga_params: GAParameters,
               movements_supplier: GAMovementsSupplier,
               objective_function: ObjectiveFunction):
    self.__ga_params: GAParameters = ga_params
    self.__movements_supplier: GAMovementsSupplier = movements_supplier
    self.__function: ObjectiveFunction = objective_function

  def run(self) -> tuple:
    population = self.__movements_supplier.create_population()
    fitness = self.__movements_supplier.compute_population_fitness(self.__function, population)

    best_individual, best_fitness = self.__movements_supplier.get_best(self.__function, population, fitness)

    population_with_fitness = list(zip(fitness, population))

    for i in range(self.__ga_params.n_generations):
      if i % 25 == 0:
        logging.warning(f'Generation: {i} - Current best: {best_fitness}')

      new_population = []
      new_fitness = []
      parents = self.__movements_supplier.select(population_with_fitness)

      while len(new_population) < self.__ga_params.population_size:
        offspring_1, offspring_2 = self.__movements_supplier.crossing(parents[0], parents[1])

        successor_1 = self.__movements_supplier.mutate(offspring_1)
        new_fitness.append(self.__function.evaluate(successor_1))
        new_population.append(successor_1)

        successor_2 = self.__movements_supplier.mutate(offspring_2)
        new_fitness.append(self.__function.evaluate(successor_2))
        new_population.append(successor_2)

      new_population_with_fitness = list(zip(new_fitness, new_population))
      population_with_fitness = self.generate_new_population(new_population_with_fitness, population_with_fitness)

      fitness, population = zip(*population_with_fitness)
      current_best_individual, current_best_fitness = self.__movements_supplier.get_best(self.__function, population,
                                                                                         fitness)

      if self.__function.compare_objective_values(current_best_fitness, best_fitness) > 0:
        best_individual = current_best_individual
        best_fitness = current_best_fitness

      if validate_termination(fitness):
        break

    return best_individual, best_fitness

  def generate_new_population(self, new_population_with_fitness, population_with_fitness):

    temp_population = new_population_with_fitness + population_with_fitness
    temp_population.sort(key=lambda x: x[0])

    return temp_population[0:self.__ga_params.population_size]
