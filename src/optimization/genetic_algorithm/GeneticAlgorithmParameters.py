class GAParameters:
  def __init__(self,
               population_size,
               n_genes,
               n_generations,
               p_cross,
               p_mutate):
    self.population_size = population_size
    self.n_genes = n_genes
    self.n_generations = n_generations
    self.p_cross = p_cross
    self.p_mutate = p_mutate
