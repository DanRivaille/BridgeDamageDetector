from src.optimization.objective_function.ObjectiveFunction import ObjectiveFunction


class TSPObjectiveFunction(ObjectiveFunction):
  def __init__(self, is_minimization: bool, distance_matrix: list[list[float]]):
    super().__init__(is_minimization)
    self.__distance_matrix = distance_matrix

  def evaluate(self, solution) -> float:
    distance = self.__distance_matrix[solution[0]][solution[-1]]

    for i in range(1, len(solution)):
      distance += self.__distance_matrix[solution[i - 1]][solution[i]]

    return distance
