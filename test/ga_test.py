from src.optimization.optimization_algorithm import OptimizationAlgorithm
from src.optimization.genetic_algorithm.movements_supplier.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters
from src.optimization.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from src.optimization.objective_function.ObjectiveFunction import ObjectiveFunction
from TSPGAMovementSupplier import TSPGAMovementSupplier
from KNPGAMovementSupplier import KNPGAMovementSupplier
from TSPObjectiveFunction import TSPObjectiveFunction
from KNPObjectiveFunction import KNPObjectiveFunction
from Instances import Instances
from KNPInstance import KnapsackInstance
from TSPInstance import TSPInstance

population_size = 200
n_generation = 2000
p_cross = 0.5
p_mutate = 0.2
knp = True


if(knp):
    knapsack_parameters: Instances = KnapsackInstance("test/KNPInstances/P07.txt")
    
    n_genes = knapsack_parameters.num_items
    ga_params: GAParameters = GAParameters(population_size, n_genes, n_generation, p_cross, p_mutate)
    knp_obj_function: ObjectiveFunction = KNPObjectiveFunction(False, knapsack_parameters.capacity, knapsack_parameters.weights, knapsack_parameters.values)
    knp_movement_supplier: GAMovementsSupplier = KNPGAMovementSupplier(ga_params)

    genetic_algorithm: OptimizationAlgorithm = GeneticAlgorithm(ga_params, knp_movement_supplier, knp_obj_function)

    best_solution, best_fitness = genetic_algorithm.run()
    print(f"Solution = {best_solution} | Fitness = {best_fitness}")
else:
    tsp_parameters: Instances = TSPInstance("test/TSPInstances/In1.txt")

    n_genes = tsp_parameters.tour_size
    ga_params: GAParameters = GAParameters(population_size, n_genes, n_generation, p_cross, p_mutate)
    tsp_obj_function: ObjectiveFunction = TSPObjectiveFunction(True, tsp_parameters.matrix)
    tsp_movement_supplier: GAMovementsSupplier = TSPGAMovementSupplier(ga_params, 3)

    genetic_algorithm: OptimizationAlgorithm = GeneticAlgorithm(ga_params, tsp_movement_supplier, tsp_obj_function)

    best_solution, best_fitness = genetic_algorithm.run()
    print(f"Solution = {best_solution} | Fitness = {best_fitness}")
