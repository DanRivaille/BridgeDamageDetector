from src.optimization import OptimizationAlgorithm
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters
from src.optimization.genetic_algorithm.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier


class GeneticAlgorithm(OptimizationAlgorithm):
  def __init__(self, ga_params: GAParameters, movements_supplier: GAMovementsSupplier):
    self.__ga_params: GAParameters = ga_params
    self.__movements_supplier: GAMovementsSupplier = movements_supplier

  def run(self):
    population = self.__movements_supplier.create_population()
    fitness = self.__movements_supplier.compute_population_fitness(population)

    best_individual, best_fitness = self.__movements_supplier.get_best(population, fitness)

    for i in range(self.__ga_params.n_generations):
      population = self.__movements_supplier.select(population, fitness)

      population = self.__movements_supplier.crossing(population)

      population = self.__movements_supplier.mutate(population)

      fitness = self.__movements_supplier.compute_population_fitness(population)

      current_best_individual, current_best_fitness = self.__movements_supplier.get_best(population, fitness)

      # TODO: generalize to consider minimization and maximization case
      if current_best_fitness < best_fitness:
        best_individual = current_best_individual
        best_fitness = current_best_fitness

    return best_individual, best_fitness
