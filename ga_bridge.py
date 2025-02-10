import os
import time
from torch.utils.data import DataLoader
from torch import save
from torch import load
from src.optimization.genetic_algorithm.GeneticAlgorithmParameters import GAParameters
from src.optimization.genetic_algorithm.movements_supplier.GeneticAlgorithmMovementsSupplier import GAMovementsSupplier
from src.optimization.optimization_algorithm import OptimizationAlgorithm
from src.optimization.genetic_algorithm.genetic_algorithm import GeneticAlgorithm
from src.optimization.objective_function.ObjectiveFunction import ObjectiveFunction
from src.damage_detector.CommonPath import CommonPath
from src.damage_detector.ConfigParams import ConfigParams
from src.damage_detector.ParserArguments import ParserArguments
from src.models.Autoencoder import Autoencoder
from src.models.CustomDataset import CustomDataset
from src.damage_detector.utils import __get_device, build_model_folder_path, load_data
from src.optimization.objective_function.BridgeObjectiveFunction import BridgeObjectiveFunction
from src.optimization.genetic_algorithm.movements_supplier.BridgeMovementsSupplier import BridgeMovementSupplier

if __name__ == '__main__':
    args = ParserArguments()

    # Load configs
    config_params = ConfigParams.load(os.path.join(CommonPath.CONFIG_FILES_FOLDER.value, args.config_filename))

    sequences_length = config_params.get_params('global_variables').get('sequences_length')
    # Load data
    train_data, validation_data = load_data(config_params, is_train=True)

    device_to_use = __get_device()
    print(device_to_use)

    # Create the model - train params
    num_epochs = config_params.get_params('train_params')['num_epochs']
    batch_size = config_params.get_params('train_params')['batch_size']
    learning_rate = config_params.get_params('train_params')['learning_rate']

    # Ga params
    to_mask = config_params.get_params('ga_params')['to_mask']
    
    n_genes = config_params.get_params('mask_n_genes')[to_mask]

    population_size = config_params.get_params('ga_params')['population_size']
    n_generations = config_params.get_params('ga_params')['n_generations']
    p_mutate = config_params.get_params('ga_params')['p_mutate']
    proportion_rate = config_params.get_params('ga_params')['proportion_rate']

    train_set = CustomDataset(train_data)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    validation_set = CustomDataset(validation_data)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)

    load_modelo = True

    if(load_modelo):
        model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), args.folder_name)
        model_path = os.path.join(model_folder, 'model_trained.pth')

        model = Autoencoder(sequences_length, to_mask)
        model.load_state_dict(load(model_path))
        model.eval()
        model.to(device_to_use)
    else:
        model = Autoencoder(sequences_length, to_mask)
        model.to(device_to_use)

    # Entrenar modelo.
    ga_params: GAParameters = GAParameters(population_size, n_genes, n_generations, p_mutate, proportion_rate)
    bridge_obj_function: ObjectiveFunction = BridgeObjectiveFunction(True, model, train_loader, validation_loader,
                                                                     learning_rate, num_epochs, device_to_use,
                                                                     proportion_rate)
    bridge_movement_supplier: GAMovementsSupplier = BridgeMovementSupplier(ga_params)

    genetic_algorithm: OptimizationAlgorithm = GeneticAlgorithm(ga_params, bridge_movement_supplier, bridge_obj_function)
    start_time = time.time()
    best_solution, best_fitness = genetic_algorithm.run()
    print(f"Tiempo de ejecucion = {time.time() - start_time}")
    print(f"Solution = {best_solution} | Fitness = {best_fitness}")

    if args.save:
        # Save the results
        model_folder = build_model_folder_path(args.model_id, config_params.get_params('id'), args.folder_name)
        os.makedirs(model_folder, exist_ok=True)

        model_path = os.path.join(model_folder, 'model_trained.pth')
        save(model.state_dict(), model_path)

        
