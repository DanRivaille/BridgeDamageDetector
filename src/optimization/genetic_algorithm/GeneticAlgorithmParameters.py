class GAParameters:
  def __init__(self,
               population_size,
               n_genes,
               n_generations,
               p_mutate,
               proportion_rate):
    self.population_size = population_size
    self.n_genes = n_genes
    self.n_generations = n_generations
    self.p_mutate = p_mutate
    self.proportion_rate = proportion_rate
