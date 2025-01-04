from src.optimization.optimization_algorithm import OptimizationAlgorithm
from src.optimization.genetic_algorithm.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters
from src.optimization.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from src.optimization.objective_function.ObjectiveFunction import ObjectiveFunction
from TSPGAMovementSupplier import TSPGAMovementSupplier
from KNPGAMovementSupplier import KNPGAMovementSupplier
from TSPObjectiveFunction import TSPObjectiveFunction
from KNPObjectiveFunction import KNPObjectiveFunction

tsp_data = {
"TourSize" : 13,
"OptTourKnow" : [3, 2, 7, 0, 9, 5, 10, 11, 1, 8, 6, 12, 4],
"OptDistanceKnow" : 7293,
"DistanceMatrix" :
    [[0, 2451,  713, 1018, 1631, 1374, 2408,  213, 2571,  875, 1420, 2145, 1972],
     [2451,    0, 1745, 1524,  831, 1240,  959, 2596,  403, 1589, 1374,  357,  579],
     [713, 1745,    0,  355,  920,  803, 1737,  851, 1858,  262,  940, 1453, 1260],
     [1018, 1524,  355,    0,  700,  862, 1395, 1123, 1584,  466, 1056, 1280,  987],
     [1631,  831,  920,  700,    0,  663, 1021, 1769,  949,  796,  879,  586,  371],
     [1374, 1240,  803,  862,  663,    0, 1681, 1551, 1765,  547,  225,  887,  999],
     [2408,  959, 1737, 1395, 1021, 1681,    0, 2493,  678, 1724, 1891, 1114,  701],
     [213, 2596,  851, 1123, 1769, 1551, 2493,    0, 2699, 1038, 1605, 2300, 2099],
     [2571,  403, 1858, 1584,  949, 1765,  678, 2699,    0, 1744, 1645,  653,  600],
     [875, 1589,  262,  466,  796,  547, 1724, 1038, 1744,    0,  679, 1272, 1162],
     [1420, 1374,  940, 1056,  879,  225, 1891, 1605, 1645,  679,    0, 1017, 1200],
     [2145,  357, 1453, 1280,  586,  887, 1114, 2300,  653, 1272, 1017,    0,  504],
     [1972,  579, 1260,  987,  371,  999,  701, 2099,  600, 1162, 1200,  504,    0]]
}

# Se busca maximizar

knp_data = {
    "capacity": 10,
    "weights": [2,3,4,5],
    "values": [3,4,5,6],
    "num_items": 4
}

knp_data2 = {
    "capacity": 14239,  
    "weights": [
        845, 758, 421, 259, 512, 405, 784, 304, 477, 584, 
        909, 505, 282, 756, 619, 251, 910, 983, 811, 903, 
        311, 730, 899, 684, 473, 101, 435, 611, 914, 967, 
        478, 866, 261, 806, 549, 15, 720, 399, 825, 669, 
        2, 494, 868, 244, 326, 871, 192, 568, 239, 968
    ],
    "values": [
        945, 858, 521, 359, 612, 505, 884, 404, 577, 684, 
        1009, 605, 382, 856, 719, 351, 1010, 1083, 911, 1003, 
        411, 830, 999, 784, 573, 201, 535, 711, 1014, 1067, 
        578, 966, 361, 906, 649, 115, 820, 499, 925, 769, 
        102, 594, 968, 344, 426, 971, 292, 668, 339, 1068
    ],
    "num_items": 50
}

population_size = 50

n_generation = 1000
p_cross = 0.8
p_mutate = 0.2

knp = True

if(knp):
    n_genes = knp_data2['num_items']
    ga_params: GAParameters = GAParameters(population_size, n_genes, n_generation, p_cross, p_mutate)
    knp_obj_function: ObjectiveFunction = KNPObjectiveFunction(False, knp_data2['capacity'], knp_data2['weights'], knp_data2['values'])
    knp_movement_supplier: GAMovementsSupplier = KNPGAMovementSupplier(ga_params, 3)

    genetic_algorithm: OptimizationAlgorithm = GeneticAlgorithm(ga_params, knp_movement_supplier, knp_obj_function)

    best_solution, best_fitness = genetic_algorithm.run()
    print(f"Solution = {best_solution} | Fitness = {best_fitness}")
else:
    n_genes = tsp_data['TourSize']
    ga_params: GAParameters = GAParameters(population_size, n_genes, n_generation, p_cross, p_mutate)
    tsp_obj_function: ObjectiveFunction = TSPObjectiveFunction(True, tsp_data['DistanceMatrix'])
    tsp_movement_supplier: GAMovementsSupplier = TSPGAMovementSupplier(ga_params, 3)

    genetic_algorithm: OptimizationAlgorithm = GeneticAlgorithm(ga_params, tsp_movement_supplier, tsp_obj_function)

    best_solution, best_fitness = genetic_algorithm.run()
    print(f"Solution = {best_solution} | Fitness = {best_fitness}")
