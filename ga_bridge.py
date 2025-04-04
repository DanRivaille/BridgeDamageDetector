import os
import time
import logging
import copy

from torch.utils.data import DataLoader
from torch.nn.utils import prune

import torch
from torch import load
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters
from src.optimization.genetic_algorithm.movements_supplier.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier
from src.optimization.optimization_algorithm import OptimizationAlgorithm
from src.optimization.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from src.optimization.objective_function.ObjectiveFunction import ObjectiveFunction
from src.damage_detector.CommonPath import CommonPath
from src.damage_detector.ConfigParams import ConfigParams
from src.damage_detector.ParserArguments import ParserArguments
from src.models.AutoencoderGA import Autoencoder
from src.models.CustomDataset import CustomDataset
from src.damage_detector.utils import __get_device, build_model_folder_path, load_data
from src.optimization.objective_function.BridgeObjectiveFunction import BridgeObjectiveFunction, evaluate_model
from src.optimization.genetic_algorithm.movements_supplier.BridgeMovementsSupplier import BridgeMovementSupplier

if __name__ == '__main__':
  args = ParserArguments()

  config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))
  sequences_length = config_params.get_params('global_variables').get('sequences_length')

  train_data, validation_data = load_data(config_params, is_train=True)

  device_to_use = __get_device()
  print(device_to_use)

  batch_size = config_params.get_params('train_params')['batch_size']
  to_mask = config_params.get_params('ga_params')['to_mask']

  n_genes = config_params.get_params('mask_n_genes')[to_mask]
  population_size = config_params.get_params('ga_params')['population_size']
  n_generations = config_params.get_params('ga_params')['n_generations']
  p_mutate = config_params.get_params('ga_params')['p_mutate']
  p_cross = config_params.get_params('ga_params')['p_cross']
  proportion_rate = config_params.get_params('ga_params')['proportion_rate']

  validation_set = CustomDataset(validation_data)
  validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

  model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), args.folder_name)
  model_path = os.path.join(model_folder, 'model_trained.pth')

  model = Autoencoder(sequences_length, to_mask)
  model.load_state_dict(load(model_path))
  model.eval()
  model.to(device_to_use)

  # Entrenar modelo.
  ga_params: GAParameters = GAParameters(population_size, n_genes, n_generations, p_mutate, p_cross, proportion_rate)
  bridge_obj_function: ObjectiveFunction = BridgeObjectiveFunction(True, model, validation_loader,
                                                                   device_to_use, proportion_rate)
  bridge_movement_supplier: GAMovementsSupplier = BridgeMovementSupplier(ga_params)

  genetic_algorithm: OptimizationAlgorithm = GeneticAlgorithm(ga_params, bridge_movement_supplier, bridge_obj_function)

  # Base fitness
  base_model_fitness = evaluate_model(model, validation_loader, device_to_use)
  logging.warning(f'Base model fitness: {base_model_fitness}')

  # Clasical pruning techniques fitness
  pruned_model_l1 = copy.deepcopy(model).to(device_to_use)
  pruned_model_l2 = copy.deepcopy(model).to(device_to_use)

  prune.ln_structured(
    module=pruned_model_l1.decoder[0],
    name='weight',
    amount=1 - proportion_rate,
    n=1,
    dim=0)

  prune.ln_structured(
    module=pruned_model_l2.decoder[0],
    name='weight',
    amount=1 - proportion_rate,
    n=2,
    dim=0)

  mean_loss_l1 = evaluate_model(pruned_model_l1, validation_loader, device_to_use)
  mean_loss_l2 = evaluate_model(pruned_model_l2, validation_loader, device_to_use)
  print(f"Best validation loss (L1) {mean_loss_l1}")
  print(f"Best validation loss (L2) {mean_loss_l2}")

  # GA fitness
  start_time = time.time()
  best_solution, best_fitness = genetic_algorithm.run()
  total_time = time.time() - start_time
  print(f"Tiempo de ejecucion = {total_time}")
  print(f"Best Fitness = {best_fitness}")
  print(best_solution)

  if args.save:
    # Save the results
    pass
