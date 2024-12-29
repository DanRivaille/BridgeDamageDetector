from src.optimization import OptimizationAlgorithm


class GeneticAlgorithm(OptimizationAlgorithm):
  def __init__(self,
               population_size,
               n_genes,
               n_generations,
               p_cross,
               p_mutate,):
    self.population_size = None
    self.n_genes = None
    self.n_generations = None
    self.p_cross = None
    self.p_mutate = None
    self.set_params(**locals())

  def set_params(self, **params):
    self.population_size = params['population_size']
    self.n_genes = params['n_genes']
    self.n_generations = params['n_generations']
    self.p_cross = params['p_cross']
    self.p_mutate = params['p_mutate']

  def run(self, **kwargs):
    population = create_population(self.n_genes, self.population_size)
    fitness = calc_fitness_population(population)

    best, min_fitness = get_best(population, fitness)

    for i in range(self.n_generations):
      population = select(population, fitness, 3, self.population_size)

      population = crossing(population, self.population_size, self.p_cross)

      population = mutate(population, self.population_size, self.p_mutate)

      fitness = calc_fitness_population(population)

      current_best, current_min_fitness = get_best(population, fitness)

      if current_min_fitness < min_fitness:
        best = current_best
        min_fitness = current_min_fitness

    return best, min_fitness
