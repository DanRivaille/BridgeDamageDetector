from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters
from src.optimization.optimization_result.OptimizationResult import OptimizationResult


class GeneticAlgorithmResult(OptimizationResult):
  def __init__(self,
               base_fitness: float,
               best_fitness: float,
               l1_fitness: float,
               l2_fitness: float,
               total_time: float,
               best_individual: list,
               ga_params: GAParameters
               ):
    self.__base_fitness = base_fitness
    self.__best_fitness = best_fitness
    self.__l1_fitness = l1_fitness
    self.__l2_fitness = l2_fitness
    self.__total_time = total_time
    self.__best_individual = best_individual
    self.__ga_params = ga_params

  def to_json(self) -> dict:
    return {
      'base_fitness': self.__base_fitness,
      'best_fitness': self.__best_fitness,
      'l1_fitness': self.__l1_fitness,
      'l2_fitness': self.__l2_fitness,
      'total_time': self.__total_time,
      'best_individual': self.__best_individual,
      'ga_params': {
        'population_size': self.__ga_params.population_size,
        'n_genes': self.__ga_params.n_genes,
        'n_generations': self.__ga_params.n_generations,
        'p_mutate': self.__ga_params.p_cross,
        'p_cross': self.__ga_params.p_cross
      },
      'proportion_rate': self.__ga_params.proportion_rate
    }
