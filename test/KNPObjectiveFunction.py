from src.optimization.objective_function.ObjectiveFunction import ObjectiveFunction


class KNPObjectiveFunction(ObjectiveFunction):
  def __init__(self, is_minimization: bool, k_capacity: int, weights: list[int], values: list[int]):
    super().__init__(is_minimization) # Deberia de ser false, es de maximizacion
    self.__k_capacity = k_capacity
    self.__weights = weights
    self.__values = values

  def evaluate(self, solution) -> int:
    weight_cont = 0
    value_cont = 0

    weight_value = list(zip(self.__weights, self.__values))
    
    for i,bit in enumerate(solution):
      if bit == 1:
        weight,value = weight_value[i]
        weight_cont += weight
        value_cont += value
    
    # Las soluciones que excedan la capacidad tendran fitness 0
    # La idea es que no incidan y no se consideren.
    if(weight_cont <= self.__k_capacity):
      return value_cont
    
    return 0
  

  def get_weight(self, solution) -> int:
    weight_cont = 0
    
    for i,bit in enumerate(solution):
      if bit == 1:
        weight = self.__weights[i]
        weight_cont += weight
        
    return weight_cont


