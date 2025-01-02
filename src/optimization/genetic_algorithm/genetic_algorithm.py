from src.optimization.optimization_algorithm import OptimizationAlgorithm
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters
from src.optimization.genetic_algorithm.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier
from src.optimization.objective_function.ObjectiveFunction import ObjectiveFunction


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

    for i in range(self.__ga_params.n_generations):
      population = self.__movements_supplier.select(population, fitness)
      population = self.__movements_supplier.crossing(population)
      population = self.__movements_supplier.mutate(population)

      fitness = self.__movements_supplier.compute_population_fitness(self.__function, population)

      current_best_individual, current_best_fitness = self.__movements_supplier.get_best(self.__function, population,
                                                                                         fitness)

      if self.__function.compare_objective_values(current_best_fitness, best_fitness) > 0:
        best_individual = current_best_individual
        best_fitness = current_best_fitness

    return best_individual, best_fitness
